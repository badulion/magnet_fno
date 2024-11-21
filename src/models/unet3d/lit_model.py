import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.data.grid import MagnetGridIterator
from torch.utils.data import DataLoader
from src.models import UNet3D

class LitUNet3D(pl.LightningModule):
    def __init__(self, model: UNet3D):
        super().__init__()
        self.model = model
        self.dataset = MagnetGridIterator(
            "../data/processed/batch_6/grid_voxel_size_4_data_type_float32", 
            phase_samples_per_simulation=100
            )
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=4, num_workers=16)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs = batch['input']
        coils = batch['coils']
        x = torch.cat((inputs, coils), dim=1)
        y = batch['field'].view((5, 12, 121, 111, 126))
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    #def validation_step(self, batch, batch_idx):
        #x, y = batch
        #y_hat = self(x)
        #val_loss = F.mse_loss(y_hat, y)
        #self.log("val_loss", val_loss)
        #return loss

    #def test_step(self, batch, batch_idx):
        #x, y = batch
        #y_hat = self(x)
        #test_loss = F.mse_loss(y_hat, y)
        #self.log("test_loss", test_loss)
        #return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
