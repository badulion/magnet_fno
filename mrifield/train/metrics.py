import einops
import torch
import time
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_msssim import SSIM

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses import MSELoss, MAELoss

from mrifield.models import UNet3D
from neuralop.models import FNO
from mrifield.train.lit_mrifield import LitMRIField
from mrifield.train.mask_padding import SubjectMaskPadding

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/fno/7hqhtfvz/checkpoints/epoch=9-step=173750.ckpt"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=64, positional_embedding=None)

train_input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/input_normalization.json")
train_target_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/target_normalization.json")
val_input_normalizer = StandardNormalizer.load_from_json(f"{VAL_DIR}/normalization/input_normalization.json")
val_target_normalizer = StandardNormalizer.load_from_json(f"{VAL_DIR}/normalization/target_normalization.json")

trained_model = LitMRIField.load_from_checkpoint(
    CKPT,
    model=model,
    train_input_normalizer=train_input_normalizer,
    train_target_normalizer=train_target_normalizer,
    val_input_normalizer=val_input_normalizer,
    val_target_normalizer=val_target_normalizer
)

trained_model.cuda()
trained_model.eval()

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        CoilEnumeratorPhaseShift(num_coils=8)
    ]
)

val_set = MagnetGridIterator(VAL_DIR, transforms=augmentation, num_samples=8)
val_loader = DataLoader(val_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

mask_padding = SubjectMaskPadding()
ssim = SSIM(data_range=1, size_average=True, channel=12)

mse = MSELoss()
mae = MAELoss()

mse_e = []
mse_h = []
mse_e_space = []
mse_h_space = []
mse_e_subject = []
mse_h_subject = []

mae_e_subject = []
mae_h_subject = []

y_hats_e_subject = []
y_hats_h_subject = []

ssim_values = []

inf_times = []

print(ModelSummary(trained_model, max_depth=-1))

for batch in tqdm(val_loader, desc="Metrics"):
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()
        subject = mask_padding(subject.unsqueeze(1)).squeeze(1)

        x = val_input_normalizer(torch.cat([inputs, coils], dim=1))

        # Measure inference time
        torch.cuda.synchronize()
        start = time.perf_counter()

        y_hat = val_target_normalizer.inverse(trained_model(x))

        torch.cuda.synchronize()
        end = time.perf_counter()
        inf_times.append((end - start) / x.shape[0])

        # Compute SSIM
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')

        y_norm = (y - y.min()) / (y.max() - y.min())
        y_hat_norm = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())

        ssim_x = ssim(y_hat_norm[:, :, 50, :, :], y_norm[:, :, 50, :, :]).cpu()
        ssim_y = ssim(y_hat_norm[:, :, :, 50, :], y_norm[:, :, :, 50, :]).cpu()
        ssim_z = ssim(y_hat_norm[:, :, :, :, 50], y_norm[:, :, :, :, 50]).cpu()
        ssim_values.append(np.mean([ssim_x, ssim_y, ssim_z]))

        # Compute MSE, MAE, and MAD
        y_e = einops.rearrange(field[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        
        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)
        y_hat_e = einops.rearrange(y_hat[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        mse_e.append(mse(y_hat_e, y_e).cpu())
        mse_h.append(mse(y_hat_h, y_h).cpu())

        mse_e_space.append(mse(y_hat_e, y_e, ~subject).cpu())
        mse_h_space.append(mse(y_hat_h, y_h, ~subject).cpu())

        mse_e_subject.append(mse(y_hat_e, y_e, subject).cpu())
        mse_h_subject.append(mse(y_hat_h, y_h, subject).cpu())

        mae_e_subject.append(mae(y_hat_e, y_e, subject).cpu())
        mae_h_subject.append(mae(y_hat_h, y_h, subject).cpu())

        y_hats_e_subject.extend(y_hat_e[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].flatten().cpu().numpy())
        y_hats_h_subject.extend(y_hat_h[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].flatten().cpu().numpy())

print(f"mse_efield: {np.mean(mse_e)}")
print(f"mse_hfield: {np.mean(mse_h)}")
print(f"mse_efield_space: {np.mean(mse_e_space)}")
print(f"mse_hfield_space: {np.mean(mse_h_space)}")
print(f"mse_efield_subject: {np.mean(mse_e_subject)}")
print(f"mse_hfield_subject: {np.mean(mse_h_subject)}")

print(f"rmse_efield_subject: {np.sqrt(np.mean(mse_e_subject))}")
print(f"rmse_hfield_subject: {np.sqrt(np.mean(mse_h_subject))}")

print(f"mae_efield_subject: {np.mean(mae_e_subject)}")
print(f"mae_hfield_subject: {np.mean(mae_h_subject)}")

print(f"mad_efield_subject: {np.median(np.abs(y_hats_e_subject - np.median(y_hats_e_subject)))}")
print(f"mad_hfield_subject: {np.median(np.abs(y_hats_h_subject - np.median(y_hats_h_subject)))}")

print(f"ssim_mean: {np.mean(ssim_values)}")

print(f"inf_median: {np.median(inf_times)}")
