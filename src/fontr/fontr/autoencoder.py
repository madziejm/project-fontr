from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy  # todo poetry add torchmetrics


class Autoencoder(pl.LightningModule):
    """TODOTODO

    Args:
        pl (_type_): _description_
    """

    def __init__(self) -> None:
        super.__init__()
        self.lr = 0.0  # todo
        self.nclasses = 0  # todo

        # todo scale input to 105x105
        # todo convert to grayscale
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Conv2d(in_channels, out_channels),  # todo
            nn.AvgPool2d(kernel_size),  # todo
            nn.Conv2d(in_channels, out_channels),  # todo
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,  # todo define the param # type: ignore[name-defined]
                out_channels,  # todo define the param # type: ignore[name-defined]
                kernel_size,  # todo define the param # type: ignore[name-defined]
            ),  # todo deconvolution
            # F.interpolate(), # ? # https://github.com/pytorch/pytorch/issues/19805
            # nn.MaxUnpool2d(),
            # another deconv
        )

        self.accuracy = Accuracy(task="multiclass", num_classes=self.nclasses)

    def encode(self, x):
        pass  # todo

    def decode(self, x):
        pass  # todo

    def forward(self, x):
        # todo scale input
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])

    def validation_step(self, batch, batch_idx):
        # self.accuracy.update(preds, y) # self.val_accuracy.update(preds, y)
        pass  # todo

    def test_step(self, batch, batch_idx):
        # self.accuracy.update(preds, y) # self.val_accuracy.update(preds, y)
        pass  # todo

    def configure_optimizers(self):
        return Adagrad(self.parameters(), lr=self.lr)
