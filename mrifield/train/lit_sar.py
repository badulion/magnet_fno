import torch
import einops

from pytorch_lightning import LightningModule

from magnet_pinn.losses import MSELoss

class LitSAR(LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 subject_lambda: float = 10.0,
                 space_lambda: float = 0.01):
        super(LitSAR, self).__init__()

        self.model = model

        self.subject_lambda = subject_lambda
        self.space_lambda = space_lambda

        self.loss_fn = MSELoss()

    def load_state_dict(self, state_dict, strict=True):
        state_dict.pop('_metadata', None)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mri, b1plus, sar, subject = batch['mri'], batch['b1plus'], batch['sar'], batch['subject']

        x = torch.cat([mri, b1plus], dim=1)
        y = sar

        y_hat = self.model(x)

        subject_loss = self.loss_fn(y_hat, y, subject)
        space_loss = self.loss_fn(y_hat, y, ~subject)

        loss = subject_loss*self.subject_lambda + space_loss*self.space_lambda

        self.log('tr_loss', loss, prog_bar=True)
        self.log('tr_subject_loss', subject_loss, prog_bar=True)
        self.log('tr_space_loss', space_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mri, b1plus, sar, subject = batch['mri'], batch['b1plus'], batch['sar'], batch['subject']

        x = torch.cat([mri, b1plus], dim=1)
        y = sar

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
