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

mse = MSELoss()
mae = MAELoss()
ssim = SSIM(data_range=1, size_average=True, channel=2)

mse_efield = []
mse_hfield = []
mse_efield_space = []
mse_hfield_space = []
mse_efield_subject = []
mse_hfield_subject = []

mae_efield_subject = []
mae_hfield_subject = []

ssim_values = []

inf_times = []

print(ModelSummary(trained_model, max_depth=-1))

for batch in tqdm(val_loader, desc="Metrics"):
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()

        x = val_input_normalizer(torch.cat([inputs, coils], dim=1))

        # Measure inference time
        torch.cuda.synchronize()
        start = time.perf_counter()

        y_hat = trained_model(x)

        torch.cuda.synchronize()
        end = time.perf_counter()
        inf_times.append((end - start) / x.shape[0])

        # Compute SSIM
        y_re = field[:, :, 0, :, :, :, :]
        y_im = field[:, :, 1, :, :, :, :]

        y_norm = torch.norm(torch.complex(y_re, y_im), dim=2)
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min())

        y_hat = einops.rearrange(val_target_normalizer.inverse(y_hat), 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)

        y_hat_re = y_hat[:, :, 0, :, :, :, :]
        y_hat_im = y_hat[:, :, 1, :, :, :, :]

        y_hat_norm = torch.norm(torch.complex(y_hat_re, y_hat_im), dim=2)
        y_hat_norm = (y_hat_norm - y_hat_norm.min()) / (y_hat_norm.max() - y_hat_norm.min())

        ssim_x = ssim(y_hat_norm[:, :, 50, :, :], y_norm[:, :, 50, :, :]).cpu()
        ssim_y = ssim(y_hat_norm[:, :, :, 50, :], y_norm[:, :, :, 50, :]).cpu()
        ssim_z = ssim(y_hat_norm[:, :, :, :, 50], y_norm[:, :, :, :, 50]).cpu()
        ssim_values.append(np.mean([ssim_x, ssim_y, ssim_z]))

        # Compute MSE and MAE
        y_e = einops.rearrange(field[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        y_hat_e = einops.rearrange(y_hat[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        mse_efield.append(mse(y_hat_e, y_e).cpu())
        mse_hfield.append(mse(y_hat_h, y_h).cpu())

        mse_efield_space.append(mse(y_hat_e, y_e, ~subject).cpu())
        mse_hfield_space.append(mse(y_hat_h, y_h, ~subject).cpu())

        mae_efield_subject.append(mae(y_hat_e, y_e, subject).cpu())
        mae_hfield_subject.append(mae(y_hat_h, y_h, subject).cpu())

        mse_efield_subject.append(mse(y_hat_e, y_e, subject).cpu())
        mse_hfield_subject.append(mse(y_hat_h, y_h, subject).cpu())

print(f"mse_efield: {np.mean(mse_efield)}")
print(f"mse_hfield: {np.mean(mse_hfield)}")
print(f"mse_efield_space: {np.mean(mse_efield_space)}")
print(f"mse_hfield_space: {np.mean(mse_hfield_space)}")
print(f"mse_efield_subject: {np.mean(mse_efield_subject)}")
print(f"mse_hfield_subject: {np.mean(mse_hfield_subject)}")

print(f"rmse_efield_subject: {np.sqrt(np.mean(mse_efield_subject))}")
print(f"rmse_hfield_subject: {np.sqrt(np.mean(mse_hfield_subject))}")

print(f"mae_efield_subject: {np.mean(mae_efield_subject)}")
print(f"mae_hfield_subject: {np.mean(mae_hfield_subject)}")

print(f"ssim_mean: {np.mean(ssim_values)}")

print(f"inf_median: {np.median(inf_times)}")
