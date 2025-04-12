import einops
import torch
import time

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_msssim import SSIM
from sklearn.metrics import r2_score

from magnet_pinn.utils import StandardNormalizer, StandardNormalizerSqrt
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses import MSELoss, MAELoss
from magnet_pinn.losses.physics import DivergenceLoss

from mrifield.models import UNet3D, AFNONet
from neuralop.models import FNO, UNO
from mrifield.train.lit_mrifield import LitMRIField

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"
TEST_DIR = "/anvme/workspace/b190cb19-magnet/processed/test/grid_voxel_size_4_data_type_float32"

BOOST_CKPT = "/home/vault/b190cb/b190cb19/fno/u0byloyd/checkpoints/epoch=14-step=260625.ckpt"
CKPT = "/home/vault/b190cb/b190cb19/fno/u0byloyd/checkpoints/epoch=14-step=260625.ckpt"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=59, positional_embedding=None)
#model = UNO(in_channels=5, out_channels=12, hidden_channels=16, n_layers=5, uno_out_channels=[32,64,128,64,32], uno_n_modes=[[13,13,13],[13,13,13],[13,13,13],[13,13,13],[13,13,13]], uno_scalings=[[1,1,1],[0.5,0.5,0.5],[1,1,1],[1,1,1],[2,2,2]], channel_mlp_skip='linear')
#model = AFNONet()

train_input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/std/input_normalization.json")
train_target_normalizer = StandardNormalizerSqrt.load_from_json(f"{TRAIN_DIR}/normalization/std/target_normalization.json")

#model_to_boost = LitMRIField.load_from_checkpoint(BOOST_CKPT, model=AFNONet(depth=5), input_normalizer=train_input_normalizer, target_normalizer=train_target_normalizer)

trained_model = LitMRIField.load_from_checkpoint(
    CKPT,
    model=model,
    input_normalizer=train_input_normalizer,
    target_normalizer=train_target_normalizer,
    #val_input_normalizer=train_input_normalizer,
    #val_target_normalizer=train_target_normalizer,
    #model_to_boost=model_to_boost
)

trained_model.cuda()
trained_model.eval()

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        CoilEnumeratorPhaseShift(num_coils=8)
    ]
)

test_set = MagnetGridIterator(TEST_DIR, transforms=augmentation, num_samples=8)
test_loader = DataLoader(test_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

ssim = SSIM(data_range=1, size_average=True, channel=12)

mse = MSELoss()
mae = MAELoss()
div = DivergenceLoss()

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

r2_e_subject = []
r2_h_subject = []

sar_subject_gt = []
sar_subject_pr = []
mse_sar_subject = []

div_subject_gt = []
div_subject_pr = []

ssim_values = []

inf_times = []

print(ModelSummary(trained_model, max_depth=-1))

for batch in tqdm(test_loader, desc="Metrics"):
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()

        x = train_input_normalizer(torch.cat([inputs, coils], dim=1))
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')

        # Measure inference time
        torch.cuda.synchronize()
        start = time.perf_counter()

        # Spectral Boost
        y_hat = torch.zeros_like(y).cuda() if trained_model.model_to_boost is None else trained_model.model_to_boost(x)

        if trained_model.model_to_boost is not None:
            x = torch.cat([x, y_hat], dim=1)

        y_hat += trained_model(x)

        torch.cuda.synchronize()
        end = time.perf_counter()
        inf_times.append((end - start) / x.shape[0])
        
        y_hat = train_target_normalizer.inverse(y_hat)

        # Compute SSIM
        y_norm = (y - y.min()) / (y.max() - y.min())
        y_hat_norm = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())

        ssim_x = ssim(y_hat_norm[:, :, 50, :, :], y_norm[:, :, 50, :, :]).cpu()
        ssim_y = ssim(y_hat_norm[:, :, :, 50, :], y_norm[:, :, :, 50, :]).cpu()
        ssim_z = ssim(y_hat_norm[:, :, :, :, 50], y_norm[:, :, :, :, 50]).cpu()
        ssim_values.append(np.mean([ssim_x, ssim_y, ssim_z]))

        # Compute MSE, MAE, MAD, and R2
        y_e = einops.rearrange(field[:, 0], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1], 'b reim xyz ... -> b (reim xyz) ...')
        
        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)
        y_hat_e = einops.rearrange(y_hat[:, 0], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1], 'b reim xyz ... -> b (reim xyz) ...')

        mse_e.append(mse(y_hat_e, y_e).cpu())
        mse_h.append(mse(y_hat_h, y_h).cpu())

        mse_e_space.append(mse(y_hat_e, y_e, ~subject).cpu())
        mse_h_space.append(mse(y_hat_h, y_h, ~subject).cpu())

        mse_e_subject.append(mse(y_hat_e, y_e, subject).cpu())
        mse_h_subject.append(mse(y_hat_h, y_h, subject).cpu())

        mae_e_subject.append(mae(y_hat_e, y_e, subject).cpu())
        mae_h_subject.append(mae(y_hat_h, y_h, subject).cpu())

        subject_exp = subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)

        y_hat_e_sub = y_hat_e[subject_exp].flatten().cpu().numpy()
        y_hat_h_sub = y_hat_h[subject_exp].flatten().cpu().numpy()

        y_e_sub = y_e[subject_exp].flatten().cpu().numpy()
        y_h_sub = y_h[subject_exp].flatten().cpu().numpy()

        y_hats_e_subject.extend(y_hat_e_sub)
        y_hats_h_subject.extend(y_hat_h_sub)

        r2_e_subject.append(r2_score(y_e_sub, y_hat_e_sub))
        r2_h_subject.append(r2_score(y_h_sub, y_hat_h_sub))

        # Compute Specific Absorption Rate (SAR)
        y_e_norm = torch.norm(field[:, 0], dim=1)
        y_hat_e_norm = torch.norm(y_hat[:, 0], dim=1)

        sigma = inputs[:, 0]
        rho = inputs[:, 2]
        
        sar_gt = sigma * torch.sum(y_e_norm**2, dim=1) / rho
        sar_pr = sigma * torch.sum(y_hat_e_norm**2, dim=1) / rho

        sar_subject_gt.append(torch.mean(sar_gt[subject]).cpu().numpy())
        sar_subject_pr.append(torch.mean(sar_pr[subject]).cpu().numpy())
        mse_sar_subject.append(mse(sar_pr, sar_gt, subject).cpu())

        # Compute divergence
        y_hat_b_re = y_hat[:,1,0]
        y_hat_b_im = y_hat[:,1,1]

        y_b_re = field[:,1,0]
        y_b_im = field[:,1,1]

        div_subject_gt.append((div(y_b_re, y_b_re, subject) + div(y_b_im, y_b_im, subject)).cpu().numpy())
        div_subject_pr.append((div(y_hat_b_re, y_b_re, subject) + div(y_hat_b_im, y_b_im, subject)).cpu().numpy())

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

print(f"r2_efield_subject: {np.mean(r2_e_subject)}")
print(f"r2_hfield_subject: {np.mean(r2_h_subject)}")

print(f"sar_subject_gt: {np.mean(sar_subject_gt)}")
print(f"sar_subject_pr: {np.mean(sar_subject_pr)}")
print(f"mse_sar_subject: {np.mean(mse_sar_subject)}")

print(f"div_subject_gt: {np.mean(div_subject_gt)}")
print(f"div_subject_pr: {np.mean(div_subject_pr)}")

print(f"ssim_mean: {np.mean(ssim_values)}")

print(f"inf_median: {np.median(inf_times)}")
