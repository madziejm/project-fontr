from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adagrad
from torchmetrics import MeanSquaredError  # todo poetry add torchmetrics


class Autoencoder(pl.LightningModule):
    """Autoencoder todo description"""

    def __init__(self, lr) -> None:
        super().__init__()
        self.lr = lr  # todo
        self.nclasses = 0  # todo

        # todo scale input to 105x105

        # todo convert to grayscale
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=48),  # todo
            nn.MaxPool2d(2),  # todo
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=24),  # todo
        )

        # todo save indices from MaxPool2d or use AvgPool2d

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,  # todo define the param # type: ignore[name-defined]
                out_channels=64,  # todo define the param # type: ignore[name-defined]
                kernel_size=24,  # todo define the param # type: ignore[name-defined]
            ),  # todo deconvolution
            nn.MaxUnpool2d(2),  # todo add indices
            # F.interpolate(), # todo alternative
            # https://github.com/pytorch/pytorch/issues/19805
            # another deconv
            nn.ConvTranspose2d(
                in_channels=64,  # todo define the param # type: ignore[name-defined]
                out_channels=3,  # todo define the param # type: ignore[name-defined]
                kernel_size=48,  # todo define the param # type: ignore[name-defined]
            ),
        )

        self.mse = nn.ModuleDict(
            {
                "train_mse": MeanSquaredError(),  # todo fix this
                "test_mse": MeanSquaredError(),  # todo fix this
                "val_mse": MeanSquaredError(),  # todo fix this
            }
        )

    def encode(self, x):
        pass  # todo

    def decode(self, x):
        pass  # todo

    def forward(self, x):
        # todo scale input
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        # todo calculate loss
        loss = 0
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])  # todo 0????

    def validation_step(self, batch, batch_idx):
        # self.mse['test_mse'].update(preds, real_y)# todo
        pass

    def test_step(self, batch, batch_idx):
        # self.mse['val_mse'].update(preds, real_y)# todo
        pass

    def configure_optimizers(self):
        return Adagrad(self.parameters(), lr=self.lr)
