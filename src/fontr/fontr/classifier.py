from typing import Any

import pytorch_lightning as pl
from torch import nn


class Classifier(pl.LightningModule):
    def __init__(
        self, lr
    ) -> None:  # todo add initializing from autoencoder pickle here
        super().__init__()
        self.lr = lr  # todo add param

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=48, stride=4),  # todo
            nn.BatchNorm2d(num_features=64),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=24),  # todo
            nn.BatchNorm2d(num_features=128),
            nn.AvgPool2d(kernel_size=2),
            # encoder ends here
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=12),  # todo
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=12),  # todo
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=12),  # todo
            nn.Linear(4096, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2383),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = 0  # TODO:
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])
