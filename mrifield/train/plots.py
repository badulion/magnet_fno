import einops
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet_pinn.utils import StandardNormalizer, StandardNormalizerSqrt
from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D, AFNONet, FNOFactorizedMesh3D
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
#model = FNOFactorizedMesh3D(modes_x=16, modes_y=16, modes_z=16, width=64, input_dim=5, output_dim=12, n_layers=8, share_weight=False, factor=4, ff_weight_norm=False, n_ff_layers=2, layer_norm=True)

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

batches = 0

rel_errs_e = np.zeros(101)
rel_errs_h = np.zeros(101)

sar10g_subject_errs = []
gt_sar = []
pr_sar = []

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
        subject_exp = subject.unsqueeze(1).expand(-1, 6, -1, -1, -1)

        x = train_input_normalizer(torch.cat([inputs, coils], dim=1))
        y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')

        y_e = einops.rearrange(field[:, 0], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1], 'b reim xyz ... -> b (reim xyz) ...')

        # Spectral Boost
        y_hat = torch.zeros_like(y).cuda() if trained_model.model_to_boost is None else trained_model.model_to_boost(x)

        if trained_model.model_to_boost is not None:
            x = torch.cat([x, y_hat], dim=1)

        y_hat += trained_model(x)
        y_hat = train_target_normalizer.inverse(y_hat)

        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)
        y_hat_e = einops.rearrange(y_hat[:, 0], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1], 'b reim xyz ... -> b (reim xyz) ...')

        # Cumulative Error Distribution
        rel_err_e = (torch.abs(y_hat_e - y_e) / torch.clamp(torch.abs(y_e), min=1e-9) * 100)[subject_exp].cpu().numpy()
        rel_err_h = (torch.abs(y_hat_h - y_h) / torch.clamp(torch.abs(y_h), min=1e-9) * 100)[subject_exp].cpu().numpy()

        for i in range(101):
            rel_errs_e[i] += np.sum(rel_err_e <= i) / len(rel_err_e)
            rel_errs_h[i] += np.sum(rel_err_h <= i) / len(rel_err_h)

        if batches <= 5:
            # Residual Histogram
            res_e.extend((y_hat_e - y_e)[subject_exp].cpu().numpy())
            res_h.extend((y_hat_h - y_h)[subject_exp].cpu().numpy())

            # Ground Truth vs. Predictions
            gt_e.extend(y_e[subject_exp].cpu().numpy())
            gt_h.extend(y_h[subject_exp].cpu().numpy())
            pr_e.extend(y_hat_e[subject_exp].cpu().numpy())
            pr_h.extend(y_hat_h[subject_exp].cpu().numpy())

        if batches <= 10:
            # SAR10g Prediction Error Histogram

            sigma = inputs[:, 0]
            rho = inputs[:, 2]

            y_e_norm = torch.norm(field[:, 0], dim=1)
            y_hat_e_norm = torch.norm(y_hat[:, 0], dim=1)

            sar_gt = sigma * torch.sum(y_e_norm**2, dim=1) / rho
            sar_pr = sigma * torch.sum(y_hat_e_norm**2, dim=1) / rho

            for voxel in torch.nonzero(subject[0]):
                for l_cube in range(1, 12, 2):
                    cube = torch.zeros_like(subject, dtype=bool)

                    cube_x = slice(max(0, voxel[0].item() - l_cube // 2), min(cube.size(dim=1), voxel[0].item() + l_cube // 2 + 1))
                    cube_y = slice(max(0, voxel[1].item() - l_cube // 2), min(cube.size(dim=2), voxel[1].item() + l_cube // 2 + 1))
                    cube_z = slice(max(0, voxel[2].item() - l_cube // 2), min(cube.size(dim=3), voxel[2].item() + l_cube // 2 + 1))

                    cube[:, cube_x, cube_y, cube_z] = True

                    if torch.sum(rho[cube & subject] * 0.004**3) >= 0.01:
                        sar10g_subject_gt = torch.mean(sar_gt[cube & subject])
                        sar10g_subject_pr = torch.mean(sar_pr[cube & subject])
                        sar10g_subject_errs.append(((sar10g_subject_pr - sar10g_subject_gt) / sar10g_subject_gt * 100).cpu().numpy())
                        if batches <= 5:
                            gt_sar.append(sar10g_subject_gt.cpu().numpy())
                            pr_sar.append(sar10g_subject_pr.cpu().numpy())

                        break

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
plt.ylabel("Fraction of Values [-]", fontsize=14)
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
plt.cla()

# SAR10g Prediction Error Histogram
plt.figure(figsize=(10, 6), dpi=300)

plt.hist(sar10g_subject_errs, bins=200, density=True, color="r", alpha=0.7)
plt.yscale("log")

plt.text(0.97, 0.97, f"μ = {np.mean(sar10g_subject_errs):.2f}\nσ = {np.std(sar10g_subject_errs):.2f}", ha="right", va="top", transform=plt.gca().transAxes)
plt.axvline(x=0, c="black", linestyle="dashed")

plt.title("SAR10g Prediction Error Histogram", fontsize=18)
plt.xlabel("Relative Error [%]", fontsize=14)
plt.ylabel("Frequency [-]", fontsize=14)

plt.savefig("./plots_hist_sar")
plt.cla()

# SAR10g Ground Truth vs. Predictions
plt.figure(figsize=(10, 6), dpi=300)

plt.scatter(gt_sar, pr_sar, s=0.5, alpha=0.05, marker=".")
plt.axline((0, 0), slope=1.0, c="black", linestyle="dashed")
plt.grid(True)

plt.title("Ground Truth vs. Predictions (SAR10g)", fontsize=18)
plt.xlabel("Ground Truth [W/kg]", fontsize=14)
plt.ylabel("Predictions [W/kg]", fontsize=14)

plt.savefig("./plots_vs_sar")
