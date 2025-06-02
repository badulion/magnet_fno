import einops
import torch
import time

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn.metrics import r2_score

from magnet_pinn.data.transforms import Compose, Crop, CoilEnumeratorPhaseShift, B1InputSARtarget
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses import MSELoss, MAELoss

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

mse = MSELoss()
mae = MAELoss()

mse_sar = []
mse_sar_space = []
mse_sar_subject = []

mae_sar_subject = []

y_hats_sar_subject = []

r2_sar_subject = []

sar_subject_gt = []
sar_subject_pr = []

inf_times = []

print(ModelSummary(trained_model, max_depth=-1))

for batch in tqdm(test_loader, desc="Metrics"):
    with torch.no_grad():
        mri, b1plus, sar, subject = batch['mri'].cuda(), batch['b1plus'].cuda(), batch['sar'].cuda(), batch['subject'].cuda()

        x = torch.cat([mri, b1plus], dim=1)
        y = sar

        # Measure inference time
        torch.cuda.synchronize()
        start = time.perf_counter()

        y_hat = trained_model(x)

        torch.cuda.synchronize()
        end = time.perf_counter()
        inf_times.append((end - start) / x.shape[0])

        # Compute MSE, MAE, MAD, and R2
        mse_sar.append(mse(y_hat, y).cpu())
        mse_sar_space.append(mse(y_hat, y, ~subject).cpu())
        mse_sar_subject.append(mse(y_hat, y, subject).cpu())
        mae_sar_subject.append(mae(y_hat, y, subject).cpu())

        subject_exp = subject.unsqueeze(1).expand(-1, 1, -1, -1, -1)
        y_hat_sub = y_hat[subject_exp].flatten().cpu().numpy()
        y_sub = y[subject_exp].flatten().cpu().numpy()

        y_hats_sar_subject.extend(y_hat_sub)

        r2_sar_subject.append(r2_score(y_sub, y_hat_sub))

        # Compute Specific Absorption Rate (SAR)
        sar_subject_gt.append(torch.mean(y[subject_exp]).cpu().numpy())
        sar_subject_pr.append(torch.mean(y_hat[subject_exp]).cpu().numpy())

print(f"mse_sar: {np.mean(mse_sar)}")
print(f"mse_sar_space: {np.mean(mse_sar_space)}")
print(f"mse_sar_subject: {np.mean(mse_sar_subject)}")

print(f"rmse_sar_subject: {np.sqrt(np.mean(mse_sar_subject))}")

print(f"mae_sar_subject: {np.mean(mae_sar_subject)}")

print(f"mad_sar_subject: {np.median(np.abs(y_hats_sar_subject - np.median(y_hats_sar_subject)))}")

print(f"r2_sar_subject: {np.mean(r2_sar_subject)}")

print(f"sar_subject_gt: {np.mean(sar_subject_gt)}")
print(f"sar_subject_pr: {np.mean(sar_subject_pr)}")

print(f"inf_median: {np.median(inf_times)}")
