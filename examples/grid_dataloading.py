from src.data.grid import MagnetGridIterator
import tqdm

iterator = MagnetGridIterator(
    "../data/processed/batch_6/grid_voxel_size_4_data_type_float32",
    phase_samples_per_simulation=100,
)

for item in tqdm.tqdm(iterator):
    pass

