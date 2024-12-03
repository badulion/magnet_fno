from magnet_pinn.preprocessing.preprocessing import GridPreprocessing
import numpy as np

preprocessor = GridPreprocessing(
    [
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_1",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_2",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_3",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_4",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_5",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_6",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_7",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_8",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_9",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_10",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_11",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_12",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_13",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_14",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_15",
        "/home/vault/b190cb/b190cb11/magnet/batches/batch_16"
    ],
    "/home/vault/b190cb/b190cb11/magnet/antenna/dipoles",
    "/anvme/workspace/b190cb19-magnet/processed/train",
    field_dtype=np.float32,
    x_min=-240,
    x_max=240,
    y_min=-220,
    y_max=220,
    z_min=-250,
    z_max=250
)

preprocessor.process_simulations()
