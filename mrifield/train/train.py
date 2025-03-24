import pytorch_lightning as pl
import wandb
import os

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from magnet_pinn.utils import StandardNormalizer, StandardNormalizerSqrt
from magnet_pinn.data.transforms import Compose, Rotate, Crop, GridPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses.physics import DivergenceLoss

from mrifield.models import UNet3D, AFNONet
from neuralop.models import FNO, UNO
from mrifield.train.lit_mrifield import LitMRIField

os.environ['HTTPS_PROXY'] = 'http://proxy:80'

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=42, positional_embedding=None)
#model = UNO(in_channels=5, out_channels=12, hidden_channels=16, uno_out_channels=[32,64,64,32], uno_n_modes=[[13,13,13],[13,13,13],[13,13,13],[13,13,13]], uno_scalings=[[1,1,1],[0.5,0.5,0.5],[1,1,1],[2,2,2]], channel_mlp_skip='linear')
#model = AFNONet()

input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/std/input_normalization.json")
target_normalizer = StandardNormalizerSqrt.load_from_json(f"{TRAIN_DIR}/normalization/std/target_normalization.json")

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100), crop_position="random"),
        GridPhaseShift(num_coils=8),
        Rotate()
    ]
)

lit_model = LitMRIField(model, input_normalizer, target_normalizer, pi_loss=DivergenceLoss())

train_set = MagnetGridIterator(TRAIN_DIR, transforms=augmentation, num_samples=100)
val_set = MagnetGridIterator(VAL_DIR, transforms=augmentation, num_samples=100)

train_loader = DataLoader(train_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

wandb_logger = WandbLogger(project="fno", name="16M - 16M, 42H, STDSQRT, DIV")
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(accelerator="gpu", devices=1, log_every_n_steps=100, max_epochs=10, logger=wandb_logger, callbacks=lr_monitor)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
