import pytorch_lightning as pl
import wandb
import os

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn

from mrifield.models import UNet3D
#from neuralop.models import FNO
from mrifield.train.lit_mrifield import LitMRIField

os.environ['HTTPS_PROXY'] = 'http://proxy:80'

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

model = UNet3D(in_channels=5, out_channels=12)
#model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=5, out_channels=12)

train_input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/input_normalization.json")
train_target_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/target_normalization.json")
val_input_normalizer = StandardNormalizer.load_from_json(f"{VAL_DIR}/normalization/input_normalization.json")
val_target_normalizer = StandardNormalizer.load_from_json(f"{VAL_DIR}/normalization/target_normalization.json")

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)

lit_model = LitMRIField(model, train_input_normalizer, train_target_normalizer, val_input_normalizer, val_target_normalizer)

train_set = MagnetGridIterator(TRAIN_DIR, transforms=augmentation, num_samples=100)
val_set = MagnetGridIterator(VAL_DIR, transforms=augmentation, num_samples=100)

train_loader = DataLoader(train_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

wandb_logger = WandbLogger(project='baseline_unet', name='Training 16M')

trainer = pl.Trainer(accelerator="gpu", devices=1, log_every_n_steps=100, max_epochs=10, logger=wandb_logger)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
