import pytorch_lightning as pl
import wandb
import os

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from magnet_pinn.utils import StandardNormalizer, StandardNormalizerSqrt
from magnet_pinn.data.transforms import Compose, Crop, GridPhaseShift, Rotate, B1InputSARtarget
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.losses.physics import DivergenceLoss, FaradaysLoss

from mrifield.models import UNet3D, AFNONet, FNOFactorizedMesh3D
from neuralop.models import FNO, UNO

from mrifield.train.lit_mrifield import LitMRIField
from mrifield.train.lit_sar import LitSAR

os.environ['HTTPS_PROXY'] = 'http://proxy:80'

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"
VAL_DIR = "/anvme/workspace/b190cb19-magnet/processed/val/grid_voxel_size_4_data_type_float32"

BOOST_CKPT = "/home/hpc/b190cb/b190cb19/ma_bohn/fno/u0byloyd/checkpoints/epoch=14-step=260625.ckpt"

### Select model architecture to train ###
#model = UNet3D(in_channels=5, out_channels=12)
model = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=59, positional_embedding=None)
#model = UNO(in_channels=5, out_channels=12, hidden_channels=16, n_layers=5, uno_out_channels=[32,64,128,64,32], uno_n_modes=[[13,13,13],[13,13,13],[13,13,13],[13,13,13],[13,13,13]], uno_scalings=[[1,1,1],[0.5,0.5,0.5],[1,1,1],[1,1,1],[2,2,2]], channel_mlp_skip='linear')
#model = AFNONet(depth=5)
#model = FNOFactorizedMesh3D(modes_x=16, modes_y=16, modes_z=16, width=64, input_dim=5, output_dim=12, n_layers=8, share_weight=False, factor=4, ff_weight_norm=False, n_ff_layers=2, layer_norm=True)

### Add model architecture to be boosted if spectral boosting is desired ###
#model_to_boost = FNO(n_modes=(16, 16, 16), in_channels=5, out_channels=12, hidden_channels=59, positional_embedding=None)

input_normalizer = StandardNormalizer.load_from_json(f"{TRAIN_DIR}/normalization/std/input_normalization.json")
target_normalizer = StandardNormalizerSqrt.load_from_json(f"{TRAIN_DIR}/normalization/std/target_normalization.json")

### Add B1InputSARtarget transform for LitSAR training ###
augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100), crop_position="random"),
        GridPhaseShift(num_coils=8),
        Rotate(),
        #B1InputSARtarget()
    ]
)

### Load trained model to be boosted from a checkpoint if desired ###
#lit_model_to_boost = LitMRIField.load_from_checkpoint(BOOST_CKPT, model=model_to_boost, input_normalizer=input_normalizer, target_normalizer=target_normalizer)

### Select training task: Prediction of E/H-field distributions or SAR maps ###
lit_model = LitMRIField(model, input_normalizer, target_normalizer)
#lit_model = LitSAR(model)

train_set = MagnetGridIterator(TRAIN_DIR, transforms=augmentation, num_samples=100)
val_set = MagnetGridIterator(VAL_DIR, transforms=augmentation, num_samples=100)

train_loader = DataLoader(train_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_set, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

wandb_logger = WandbLogger(project="fno", name="32M - 16M, 59H")
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(accelerator="gpu", devices=1, log_every_n_steps=100, max_epochs=15, logger=wandb_logger, callbacks=lr_monitor)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
