import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.utils import StandardNormalizer
from src.data.grid import MagnetGridIterator
from src.data.utils import worker_init_fn

class LitMRIField(pl.LightningModule):
    def __init__(self, data_path: str,
                 model: torch.nn.Module,
                 input_normalizer: StandardNormalizer,
                 target_normalizer: StandardNormalizer,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01):
        super(LitMRIField, self).__init__()
        
        self.dataset = MagnetGridIterator(data_path, phase_samples_per_simulation=100)
        self.model = model
        self.input_normalizer=input_normalizer
        self.target_normalizer=target_normalizer
        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda

        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, coils, field, subject = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = self.input_normalizer(torch.cat([inputs, coils], dim=1))
        y = self.target_normalizer(field.view((4, 12, 121, 111, 126)))

        y_hat = self.model(x)

        mse = torch.mean((y_hat - y) ** 2, dim=1)
        subject_loss = torch.mean(mse * subject)
        space_loss = torch.mean(mse * (~subject))
        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log('train_loss', loss, prog_bar=True)
        self.log('subject_loss', subject_loss, prog_bar=True)
        self.log('space_loss', space_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
