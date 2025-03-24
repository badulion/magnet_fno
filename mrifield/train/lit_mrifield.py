import torch
import einops

from pytorch_lightning import LightningModule

from magnet_pinn.utils import Normalizer
from magnet_pinn.losses import MSELoss
from magnet_pinn.losses.physics import BasePhysicsLoss, DivergenceLoss

class LitMRIField(LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 input_normalizer: Normalizer,
                 target_normalizer: Normalizer,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01,
                 model_to_boost: LightningModule = None,
                 pi_loss: BasePhysicsLoss = None):
        super(LitMRIField, self).__init__()

        self.model = model

        self.input_normalizer=input_normalizer
        self.target_normalizer=target_normalizer

        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda

        self.model_to_boost = model_to_boost
        self.pi_loss = pi_loss

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

        # Spectral Boost
        if self.model_to_boost is not None:
            y_hat_to_boost = self.model_to_boost(x)

            x = torch.cat([x, y_hat_to_boost], dim=1)
            y = y - y_hat_to_boost

        y_hat = self.model(x)

        subject_loss = self.loss_fn(y_hat, y, subject)
        space_loss = self.loss_fn(y_hat, y, ~subject)

        # Physics-Informed Loss
        if isinstance(self.pi_loss, DivergenceLoss):
            y_hat_denorm = einops.rearrange(self.target_normalizer.inverse(y_hat), 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)
            y_denorm = einops.rearrange(self.target_normalizer.inverse(y), 'b (he reim xyz) ... -> b he reim xyz ...', he=2, reim=2, xyz=3)

            y_hat_b_re = y_hat_denorm[:,1,0]
            y_hat_b_im = y_hat_denorm[:,1,1]

            y_b_re = y_denorm[:,1,0]
            y_b_im = y_denorm[:,1,1]

            subject_loss += 1000 * (self.pi_loss(y_hat_b_re, y_b_re, subject) + self.pi_loss(y_hat_b_im, y_b_im, subject))
            space_loss += 1000 * (self.pi_loss(y_hat_b_re, y_b_re, ~subject) + self.pi_loss(y_hat_b_im, y_b_im, ~subject))

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
