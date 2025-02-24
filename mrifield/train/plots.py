import einops
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D
from neuralop.models import FNO, UNO
from mrifield.train.lit_mrifield import LitMRIField
from mrifield.train.mask_padding import SubjectMaskPadding

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/fno/7hqhtfvz/checkpoints/epoch=9-step=173750.ckpt"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=64, positional_embedding=None)
#model = UNO(in_channels=5, out_channels=12, hidden_channels=16, uno_out_channels=[32,64,64,32], uno_n_modes=[[16,16,16],[16,16,16],[16,16,16],[16,16,16]], uno_scalings=[[1,1,1],[0.5,0.5,0.5],[1,1,1],[2,2,2]], channel_mlp_skip='linear')

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
val_loader = DataLoader(val_set, batch_size=8, num_workers=16, worker_init_fn=worker_init_fn)

batches = 0
mask_padding = SubjectMaskPadding()

err_e = np.zeros(101)
err_h = np.zeros(101)

res_e = []
res_h = []

gt_e = []
pr_e = []
gt_h = []
pr_h = []

for batch in tqdm(val_loader):
    batches += 1
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()
        subject = mask_padding(subject.unsqueeze(1)).squeeze(1)

        x = val_input_normalizer(torch.cat([inputs, coils], dim=1))

        y_e = einops.rearrange(field[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        y_hat = val_target_normalizer.inverse(trained_model(x))
        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)

        y_hat_e = einops.rearrange(y_hat[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        batch_err_efield = torch.abs(y_hat_e - y_e) / torch.clamp(torch.abs(y_e), min=1e-9) * 100
        batch_err_hfield = torch.abs(y_hat_h - y_h) / torch.clamp(torch.abs(y_h), min=1e-9) * 100

        batch_err_efield = batch_err_efield[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy()
        batch_err_hfield = batch_err_hfield[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy()

        res_e = (y_hat_e - y_e)[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy()
        res_h = (y_hat_h - y_h)[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy()

        for i in range(101):
            err_e[i] += np.sum(batch_err_efield <= i) / len(batch_err_efield)
            err_h[i] += np.sum(batch_err_hfield <= i) / len(batch_err_hfield)

        gt_e = (y_e[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())
        pr_e = (y_hat_e[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())
        gt_h = (y_h[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())
        pr_h = (y_hat_h[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())

err_e /= batches
err_h /= batches

_, (cdf, hist_e, hist_h, vs_e, vs_h) = plt.subplots(1, 5, figsize=(34, 6))

cdf.plot(np.arange(0, 101), err_e, "r-", label="E-field (Subject)")
cdf.plot(np.arange(0, 101), err_h, "b-", label="H-field (Subject)")

cdf.set_xlim(0, 100)
cdf.set_ylim(0, 1)
cdf.grid(True)

cdf.set_title("Cumulative Error Distribution")
cdf.set_xlabel("Cumulative Error [%]")
cdf.set_ylabel("Fraction of Voxels [-]")
cdf.legend()

hist_e.hist(res_e, bins=100, density=True, color="r")
hist_h.hist(res_h, bins=100, density=True, color="b")

hist_e.set_yscale("log")
hist_h.set_yscale("log")

hist_e.text(0.97, 0.97, f"μ = {np.mean(res_e):.3f}\nσ = {np.std(res_e):.3f}", ha="right", va="top", transform=hist_e.transAxes)
hist_h.text(0.97, 0.97, f"μ = {np.mean(res_h):.3f}\nσ = {np.std(res_h):.3f}", ha="right", va="top", transform=hist_h.transAxes)

hist_e.axvline(x=0, linestyle="dashed")
hist_h.axvline(x=0, linestyle="dashed")

hist_e.set_title("Residual Histogram (E-field)")
hist_e.set_xlabel("Residual [-]")
hist_e.set_ylabel("Frequency [-]")

hist_h.set_title("Residual Histogram (H-field)")
hist_h.set_xlabel("Residual [-]")
hist_h.set_ylabel("Frequency [-]")

vs_e.scatter(gt_e, pr_e, s=0.5, marker=".")
vs_h.scatter(gt_h, pr_h, s=0.5, marker=".")

vs_e.plot([0, 1], [0, 1], c="black", linestyle="dashed", transform=vs_e.transAxes)
vs_h.plot([0, 1], [0, 1], c="black", linestyle="dashed", transform=vs_h.transAxes)

vs_e.set_title("Ground Truth vs. Predictions (E-field)")
vs_e.set_xlabel("Ground Truth [-]")
vs_e.set_ylabel("Predictions [-]")

vs_h.set_title("Ground Truth vs. Predictions (H-field)")
vs_h.set_xlabel("Ground Truth [-]")
vs_h.set_ylabel("Predictions [-]")

plt.savefig("./plots")
