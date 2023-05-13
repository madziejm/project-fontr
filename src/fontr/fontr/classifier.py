from typing import Any

import pytorch_lightning as pl
from torch import nn


class Classifier(pl.LightningModule):
    def __init__(
        self, lr
    ) -> None:  # todo add initializing from autoencoder pickle here
        super().__init__()
        self.lr = lr  # todo use param

        stride = 2

        self.encoder_prefix = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=stride,
                padding=2,
            ),
            nn.MaxPool2d(2, padding=1),
            # nn.BatchNorm2d(num_features=64), # todo readd me?
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=stride,
                padding=2,
            ),
            nn.MaxPool2d(2, padding=1),
        )
        # todo initialize encoder prefix from an already trained autoencoder
        # freeze encoder layers # todo is needed???
        for param in self.encoder_prefix.parameters():
            param.requires_grad = False

        self.classifier_layers = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 150),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.encoder_prefix(x)
        for layer in self.classifier_layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = 0  # TODO:
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])
