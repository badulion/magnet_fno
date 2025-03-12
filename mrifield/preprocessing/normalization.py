from magnet_pinn.utils import arcsinhStandardNormalizer
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose

import numpy as np
import einops

class Iterator:
    def __init__(self, path):
        self.path = path
        augmentation = Compose(
            [
                Crop(crop_size=(100, 100, 100)),
                GridPhaseShift(num_coils=8)
            ]
        )
    
        self.iterator = MagnetGridIterator(
            path,
            transforms=augmentation,
            num_samples=1
        )

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            input = np.concatenate([batch['input'], batch['coils']], axis=0)
            target = einops.rearrange(batch['field'], 'he reim xyz ... -> (he reim xyz) ...')
            yield {
                'input': input,
                'target': target,
            }

TRAIN_DIR = "/anvme/workspace/b190cb19-magnet/processed/train/grid_voxel_size_4_data_type_float32"

normalizer = arcsinhStandardNormalizer()

normalizer.fit_params(Iterator(TRAIN_DIR), key='input', axis=0)
normalizer.save_as_json(f"{TRAIN_DIR}/normalization/input_normalization.json")

normalizer.fit_params(Iterator(TRAIN_DIR), key='target', axis=0)
normalizer.save_as_json(f"{TRAIN_DIR}/normalization/target_normalization.json")
