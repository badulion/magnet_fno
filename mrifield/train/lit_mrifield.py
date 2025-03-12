import torch
import pytorch_lightning as pl
import einops

from magnet_pinn.utils import StandardNormalizer, arcsinhStandardNormalizer
from magnet_pinn.losses import MSELoss

class LitMRIField(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 input_normalizer: StandardNormalizer,
                 target_normalizer: arcsinhStandardNormalizer,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01):
        super(LitMRIField, self).__init__()
        
        self.model = model

        self.input_normalizer=input_normalizer
        self.target_normalizer=target_normalizer

        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda

        self.loss_fn = MSELoss()

    def load_state_dict(self, state_dict, strict=True):
        state_dict.pop('_metadata', None)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, coils, field, subject = batch['input'], batch['coils'], batch['field'], batch['subject']

        x = self.input_normalizer(torch.cat([inputs, coils], dim=1))
        y = self.target_normalizer(einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...'))

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

        x = self.input_normalizer(torch.cat([inputs, coils], dim=1))
        y = self.target_normalizer(einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...'))

        y_hat = self.model(x)

        subject_loss = self.loss_fn(y_hat, y, subject)
        space_loss = self.loss_fn(y_hat, y, ~subject)
        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log("val_loss", loss)
        self.log('val_subject_loss', subject_loss, prog_bar=True)
        self.log('val_space_loss', space_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=260625, eta_min=5e-5)

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr_scheduler"
        }
    }
