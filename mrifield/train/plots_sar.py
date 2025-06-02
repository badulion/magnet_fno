import einops
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift, B1InputSARtarget
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D
from neuralop.models import FNO

from mrifield.train.lit_sar import LitSAR

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"
TEST_DIR = "/anvme/workspace/b190cb19-magnet/processed/test/grid_voxel_size_4_data_type_float32"

CKPT = "/home/vault/b190cb/b190cb19/b1+/l87sylvh/checkpoints/epoch=14-step=260625.ckpt"

#model = UNet3D(in_channels=3, out_channels=1, f_maps=90, num_groups=9)
model = FNO(n_modes=(16, 16, 16), in_channels=3, out_channels=1, hidden_channels=59, positional_embedding=None)

trained_model = LitSAR.load_from_checkpoint(CKPT, model=model)

trained_model.cuda()
trained_model.eval()

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        CoilEnumeratorPhaseShift(num_coils=8),
        B1InputSARtarget()
    ]
)

test_set = MagnetGridIterator(TEST_DIR, transforms=augmentation, num_samples=8)
test_loader = DataLoader(test_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

batches = 0

rel_errs = np.zeros(101)

sar10g_subject_errs = []
gt_sar = []
pr_sar = []

gt_sar = []
pr_sar = []

for batch in tqdm(test_loader):
    batches += 1
    with torch.no_grad():
        mri, b1plus, sar, inputs, subject = batch['mri'].cuda(), batch['b1plus'].cuda(), batch['sar'].cuda(), batch['input'].cuda(), batch['subject'].cuda()
        subject_exp = subject.unsqueeze(1).expand(-1, 1, -1, -1, -1)

        x = torch.cat([mri, b1plus], dim=1)
        y = sar

        y_hat = trained_model(x)

        # Cumulative Error Distribution
        rel_err = (torch.abs(y_hat - y) / torch.clamp(torch.abs(y), min=1e-9) * 100)[subject_exp].cpu().numpy()

        for i in range(101):
            rel_errs[i] += np.sum(rel_err <= i) / len(rel_err)

        if batches <= 10:
            # SAR10g Prediction Error Histogram
            rho = inputs[:, 2]

            sar_gt = y[:, 0]
            sar_pr = y_hat[:, 0]

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

rel_errs /= batches

# Cumulative Error Distribution
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(np.arange(0, 101), rel_errs, "r-")

plt.xlim(0, 100)
plt.ylim(0, 1)
plt.grid(True)

plt.title("Cumulative Error Distribution", fontsize=18)
plt.xlabel("Cumulative Error [%]", fontsize=14)
plt.ylabel("Fraction of Values [-]", fontsize=14)
plt.legend()

plt.savefig("./plots_cdf")
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
