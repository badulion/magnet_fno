import einops
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.transforms import Compose, Crop, SingleCoilZeroPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D
from neuralop.models import FNO
from mrifield.train.lit_mrifield import LitMRIField

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/fno/72h9x50x/checkpoints/epoch=9-step=173750.ckpt"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=42, positional_embedding=None)

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
        SingleCoilZeroPhaseShift(num_coils=8)
    ]
)

val_set = MagnetGridIterator(VAL_DIR, transforms=augmentation, num_samples=8)
val_loader = DataLoader(val_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

batches = 0
mse_efield = 0.0
mse_hfield = 0.0
mse_efield_space = 0.0
mse_hfield_space = 0.0
mse_efield_subject = 0.0
mse_hfield_subject = 0.0

for batch in tqdm(val_loader, desc="Inference"):
    with torch.no_grad():
        batches += 1
        inputs, coils, field, subject = batch['input'].cuda(), batch['coils'].cuda(), batch['field'].cuda(), batch['subject'].cuda()

        x = val_input_normalizer(torch.cat([inputs, coils], dim=1))

        y_e = einops.rearrange(field[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_h = einops.rearrange(field[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        y_hat = val_target_normalizer.inverse(trained_model(x))
        y_hat = einops.rearrange(y_hat, 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)

        y_hat_e = einops.rearrange(y_hat[:, 0, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')
        y_hat_h = einops.rearrange(y_hat[:, 1, :, :, :, :, :], 'b reim xyz ... -> b (reim xyz) ...')

        mse_efield += trained_model.loss_fn(y_hat_e, y_e)
        mse_hfield += trained_model.loss_fn(y_hat_h, y_h)

        mse_efield_space += trained_model.loss_fn(y_hat_e, y_e, ~subject)
        mse_hfield_space += trained_model.loss_fn(y_hat_h, y_h, ~subject)

        mse_efield_subject += trained_model.loss_fn(y_hat_e, y_e, subject)
        mse_hfield_subject += trained_model.loss_fn(y_hat_h, y_h, subject)

print(f"mse_efield: {mse_efield / batches}")
print(f"mse_hfield: {mse_hfield / batches}")
print(f"mse_efield_space: {mse_efield_space / batches}")
print(f"mse_hfield_space: {mse_hfield_space / batches}")
print(f"mse_efield_subject: {mse_efield_subject / batches}")
print(f"mse_hfield_subject: {mse_hfield_subject / batches}")
