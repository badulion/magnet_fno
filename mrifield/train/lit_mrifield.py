import torch
import pytorch_lightning as pl
import einops

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.losses import MSELoss

class LitMRIField(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 train_input_normalizer: StandardNormalizer,
                 train_target_normalizer: StandardNormalizer,
                 val_input_normalizer: StandardNormalizer,
                 val_target_normalizer: StandardNormalizer,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01):
        super(LitMRIField, self).__init__()
        
        self.model = model

        self.train_input_normalizer=train_input_normalizer
        self.train_target_normalizer=train_target_normalizer
        self.val_input_normalizer=val_input_normalizer
        self.val_target_normalizer=val_target_normalizer
        
        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda
        self.loss_fn = MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, coils, field, subject = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = self.train_input_normalizer(torch.cat([inputs, coils], dim=1))
        y = self.train_target_normalizer(einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...'))

        y_hat = self.model(x)

        subject_loss = self.loss_fn(y_hat, y, subject)
        space_loss = self.loss_fn(y_hat, y, ~subject)
        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log('tr_loss', loss, prog_bar=True)
        self.log('tr_subject_loss', subject_loss, prog_bar=True)
        self.log('tr_space_loss', space_loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, coils, field, subject = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = self.val_input_normalizer(torch.cat([inputs, coils], dim=1))
        y = self.val_target_normalizer(einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...'))

        y_hat = self.model(x)

        subject_loss = self.loss_fn(y_hat, y, subject)
        space_loss = self.loss_fn(y_hat, y, ~subject)
        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log("val_loss", loss)
        self.log('val_subject_loss', subject_loss, prog_bar=True)
        self.log('val_space_loss', space_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
