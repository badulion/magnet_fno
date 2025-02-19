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
from neuralop.models import FNO
from mrifield.train.lit_mrifield import LitMRIField

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/baseline_unet/tkqjcw1e/checkpoints/epoch=9-step=173750.ckpt"

model = UNet3D(in_channels=5, out_channels=12)
#model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=42, positional_embedding=None)

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

batches = 0

err_e = np.zeros(101)
err_h = np.zeros(101)

res_e = []
res_h = []

for batch in tqdm(val_loader):
    batches += 1
    with torch.no_grad():
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()

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

        res_e.extend((y_hat_e - y_e)[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())
        res_h.extend((y_hat_h - y_h)[subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)].cpu().numpy())

        for i in range(101):
            err_e[i] += np.sum(batch_err_efield <= i) / len(batch_err_efield)
            err_h[i] += np.sum(batch_err_hfield <= i) / len(batch_err_hfield)

err_e /= batches
err_h /= batches

_, (cdf, hist_e, hist_h) = plt.subplots(1, 3, figsize=(20, 6))

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

plt.savefig("./plots")
