import einops
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet_pinn.utils import StandardNormalizer, arcsinhStandardNormalizer
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D
from neuralop.models import FNO, UNO
from mrifield.train.lit_mrifield import LitMRIField

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"
TEST_DIR = "/anvme/workspace/b190cb19-magnet/processed/test/grid_voxel_size_4_data_type_float32"

CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/fno/7hqhtfvz/checkpoints/epoch=9-step=173750.ckpt"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=64, positional_embedding=None)
#model = UNO(in_channels=5, out_channels=12, hidden_channels=16, uno_out_channels=[32,64,64,32], uno_n_modes=[[16,16,16],[16,16,16],[16,16,16],[16,16,16]], uno_scalings=[[1,1,1],[0.5,0.5,0.5],[1,1,1],[2,2,2]], channel_mlp_skip='linear')

train_input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/input_normalization.json")
train_target_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/target_normalization.json")

trained_model = LitMRIField.load_from_checkpoint(
    CKPT,
    model=model,
    train_input_normalizer=train_input_normalizer,
    train_target_normalizer=train_target_normalizer,
    #val_input_normalizer=train_input_normalizer,
    #val_target_normalizer=train_target_normalizer
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

batches = 0

rel_errs_e = np.zeros(101)
rel_errs_h = np.zeros(101)

res_e = []
res_h = []

gt_e = []
pr_e = []
gt_h = []
pr_h = []

for batch in tqdm(test_loader):
    batches += 1
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()
        subject = subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)

        x = train_input_normalizer(torch.cat([inputs, coils], dim=1))

        y_e = einops.rearrange(field[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        y_hat = train_target_normalizer.inverse(trained_model(x))
        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)

        y_hat_e = einops.rearrange(y_hat[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        rel_err_e = (torch.abs(y_hat_e - y_e) / torch.clamp(torch.abs(y_e), min=1e-9) * 100)[subject].cpu().numpy()
        rel_err_h = (torch.abs(y_hat_h - y_h) / torch.clamp(torch.abs(y_h), min=1e-9) * 100)[subject].cpu().numpy()

        for i in range(101):
            rel_errs_e[i] += np.sum(rel_err_e <= i) / len(rel_err_e)
            rel_errs_h[i] += np.sum(rel_err_h <= i) / len(rel_err_h)

        if batches < 5:
            res_e.extend((y_hat_e - y_e)[subject].cpu().numpy())
            res_h.extend((y_hat_h - y_h)[subject].cpu().numpy())
            
            gt_e.extend(y_e[subject].cpu().numpy())
            gt_h.extend(y_h[subject].cpu().numpy())
            pr_e.extend(y_hat_e[subject].cpu().numpy())
            pr_h.extend(y_hat_h[subject].cpu().numpy())

rel_errs_e /= batches
rel_errs_h /= batches

# Cumulative Error Distribution
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(np.arange(0, 101), rel_errs_e, "r-", label="E-field (Subject)")
plt.plot(np.arange(0, 101), rel_errs_h, "b-", label="H-field (Subject)")

plt.xlim(0, 100)
plt.ylim(0, 1)
plt.grid(True)

plt.title("Cumulative Error Distribution", fontsize=18)
plt.xlabel("Cumulative Error [%]", fontsize=14)
plt.ylabel("Fraction of Voxels [-]", fontsize=14)
plt.legend()

plt.savefig("./plots_cdf")
plt.cla()

# Residual Histogram
_, (hist_e, hist_h) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

hist_e.hist(res_e, bins=100, density=True, color="r", alpha=0.7)
hist_h.hist(res_h, bins=100, density=True, color="b", alpha=0.7)

hist_e.set_yscale("log")
hist_h.set_yscale("log")

hist_e.text(0.97, 0.97, f"μ = {np.mean(res_e):.3f}\nσ = {np.std(res_e):.3f}", ha="right", va="top", transform=hist_e.transAxes)
hist_h.text(0.97, 0.97, f"μ = {np.mean(res_h):.3f}\nσ = {np.std(res_h):.3f}", ha="right", va="top", transform=hist_h.transAxes)

hist_e.axvline(x=0, c="black", linestyle="dashed")
hist_h.axvline(x=0, c="black", linestyle="dashed")

hist_e.set_title("Residual Histogram (E-field)", fontsize=18)
hist_e.set_xlabel("Residual [V/m]", fontsize=14)
hist_e.set_ylabel("Frequency [-]", fontsize=14)

hist_h.set_title("Residual Histogram (H-field)", fontsize=18)
hist_h.set_xlabel("Residual [A/m]", fontsize=14)
hist_h.set_ylabel("Frequency [-]", fontsize=14)

plt.savefig("./plots_hist")
plt.cla()

# Ground Truth vs. Predictions
_, (vs_e, vs_h) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

vs_e.scatter(gt_e, pr_e, s=0.5, alpha=0.05, marker=".")
vs_h.scatter(gt_h, pr_h, s=0.5, alpha=0.05, marker=".")

vs_e.axline((0, 0), slope=1.0, c="black", linestyle="dashed")
vs_h.axline((0, 0), slope=1.0, c="black", linestyle="dashed")
vs_e.grid(True)
vs_h.grid(True)

vs_e.set_title("Ground Truth vs. Predictions (E-field)", fontsize=18)
vs_e.set_xlabel("Ground Truth [V/m]", fontsize=14)
vs_e.set_ylabel("Predictions [V/m]", fontsize=14)

vs_h.set_title("Ground Truth vs. Predictions (H-field)", fontsize=18)
vs_h.set_xlabel("Ground Truth [A/m]", fontsize=14)
vs_h.set_ylabel("Predictions [A/m]", fontsize=14)

plt.savefig("./plots_vs")
