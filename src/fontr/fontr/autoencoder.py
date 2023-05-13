from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanSquaredError  # todo poetry add torchmetrics???


class Autoencoder(pl.LightningModule):
    """Autoencoder todo description"""

    def __init__(self, lr) -> None:
        super().__init__()
        self.lr = lr  # todo
        self.nclasses = 0  # todo

        stride = 2

        # todo convert to grayscale
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=1, out_channels=64, kernel_size=3, stride=stride
                        ),
                    ),
                    ("lr1", nn.LeakyReLU(0.2)),
                    ("maxpool1", nn.MaxPool2d(2, return_indices=True)),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=stride,
                        ),
                    ),
                    ("lr2", nn.LeakyReLU(0.2)),
                ]
            )
        )

        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "convt1",
                        nn.ConvTranspose2d(
                            in_channels=128,
                            out_channels=64,
                            kernel_size=3,
                            stride=stride,
                        ),
                    ),
                    (
                        "maxunpool1",
                        nn.MaxUnpool2d(2),
                    ),  # todo add indices
                    (
                        "convt2",
                        nn.ConvTranspose2d(
                            in_channels=64, out_channels=1, kernel_size=3, stride=stride
                        ),
                    ),
                ]
            )
        )

        self.mse = nn.ModuleDict(
            {
                "train_mse": MeanSquaredError(),  # todo fix this
                "test_mse": MeanSquaredError(),  # todo fix this
                "val_mse": MeanSquaredError(),  # todo fix this
            }
        )

    def encode(self, x):
        pooling_indices = []
        encoded_x = x
        for name, layer in self.encoder.named_children():
            if name.startswith("conv") or name.startswith("lr"):
                encoded_x = layer.forward(encoded_x)
            elif name.startswith("maxpool"):
                encoded_x, layer_chosen_indices = layer.forward(encoded_x)
                pooling_indices.append(layer_chosen_indices)
            else:
                raise ValueError("Unexpected layer name")
        return encoded_x, pooling_indices

    def decode(self, encoded_x, pooling_indices):
        decoded_x = encoded_x

        for name, layer in self.decoder.named_children():
            if name.startswith("convt"):
                decoded_x = layer.forward(decoded_x)
            elif name.startswith("maxunpool"):
                decoded_x = layer.forward(
                    decoded_x, pooling_indices.pop(), output_size=(47, 47)
                )
            else:
                raise ValueError("Unexpected layer name")

        return decoded_x

    def forward(self, x):
        # todo scale input
        encoded_x, pooling_indices = self.encode(x)
        decoded_x = self.decode(encoded_x, pooling_indices)
        return decoded_x

    def training_step(self, batch, batch_idx):
        x, y = batch  # todo no x, y
        # todo calculate loss
        loss = 0
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])  # todo 0????

    def validation_step(self, batch, batch_idx):
        # self.mse['test_mse'].update(preds, real_y)2# todo
        pass

    def test_step(self, batch, batch_idx):
        # self.mse['val_mse'].update(preds, real_y)# todo
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
