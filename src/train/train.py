import pytorch_lightning as pl
from magnet_pinn.utils import StandardNormalizer
from src.models import UNet3D
from src.train.lit_mrifield import LitMRIField

#BASE_DIR = "../data/processed/batch_6/grid_voxel_size_4_data_type_float32"
BASE_DIR = "/anvme/workspace/b190cb11-magnet/data/processed/train/grid_voxel_size_4_data_type_float32"

model = UNet3D(in_channels=5, out_channels=12)

input_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/input_normalization.json")
target_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/target_normalization.json")

lit_model = LitMRIField(BASE_DIR, model, input_normalizer, target_normalizer)

trainer = pl.Trainer(accelerator="gpu", devices=1, log_every_n_steps=5, max_epochs=20)
trainer.fit(model=lit_model)
