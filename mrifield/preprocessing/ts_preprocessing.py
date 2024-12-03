from magnet_pinn.preprocessing.preprocessing import GridPreprocessing
import numpy as np

preprocessor = GridPreprocessing(
    [
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_19",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_20"
    ],
    "/home/vault/b190cb/b190cb11/magnet/antenna/dipoles",
    "/anvme/workspace/b190cb19-magnet/processed/test",
    field_dtype=np.float32,
    x_min=-240,
    x_max=240,
    y_min=-220,
    y_max=220,
    z_min=-250,
    z_max=250
)

preprocessor.process_simulations()
