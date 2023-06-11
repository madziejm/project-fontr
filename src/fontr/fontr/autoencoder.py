from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanSquaredError


class Autoencoder(pl.LightningModule):
    """Autoencoder based on the Autoencoder described in
    "DeepFont: Identify Your Font from An Image" paper"""

    def __init__(self, lr: float) -> None:
        """Construct the Autoencoder object

        Args:
            lr (float): learning rate
        """
        super().__init__()
        self.lr = lr

        # TODO convert to grayscale? TODO remove this TODO as it will be handled by the dataloader # noqa: E501
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, return_indices=True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(num_features=128),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, return_indices=True),
            ],
        )

        self.decoder = nn.ModuleList(
            [
                nn.MaxUnpool2d(2, stride=2),
                nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.MaxUnpool2d(2, stride=2),
                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ],
        )

        self.mse = nn.ModuleDict(
            {
                "train_mse": MeanSquaredError(),
                "test_mse": MeanSquaredError(),
                "val_mse": MeanSquaredError(),
            }
        )

        # this corresponds to sizes of outputs of the two Conv2d layers
        # used by MaxUnpool2d layers
        self.maxunpool_output_sizes = ([48, 48], [24, 24])

        self.loss = nn.MSELoss(reduction="mean")
        self.save_hyperparameters()

    def encode(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pooling_indices: List[torch.Tensor] = []
        encoded_x = x

        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                encoded_x, layer_chosen_indices = layer.forward(encoded_x)
                pooling_indices.append(layer_chosen_indices)
            else:
                encoded_x = layer.forward(encoded_x)
        return encoded_x, pooling_indices

    def decode(self, encoded_x, pooling_indices):
        decoded_x = encoded_x

        maxunpool_i = 1
        for layer in self.decoder:
            if isinstance(layer, nn.MaxUnpool2d):
                decoded_x = layer.forward(
                    decoded_x,
                    pooling_indices.pop(),
                    output_size=self.maxunpool_output_sizes[-maxunpool_i],
                )
                maxunpool_i += 1
            else:
                decoded_x = layer.forward(decoded_x)

        return decoded_x

    def forward(self, x):
        encoded_x, pooling_indices = self.encode(x)
        decoded_x = self.decode(encoded_x, pooling_indices)
        return decoded_x

    # def on_train_start(self) -> None:
    #     # actually logging strings is not supported in pl
    #     # TODO adjust or remove
    #     super().on_train_start()
    #     assert self.logger is not None
    #     self.log("encoder", str(self.encoder))
    #     self.log("decoder", str(self.decoder))

    def base_step(self, batch, batch_idx: int, step_name: str):
        b, p, c, h, w = batch.shape
        flatten_batch = batch.view(b * p, c, h, w)
        autoenc_output = self.forward(flatten_batch)
        autoenc_output = autoenc_output.view(b, p, c, h, w)

        mse_id: str = step_name + "_mse"
        self.mse[mse_id](autoenc_output, batch)
        self.log(
            mse_id,
            self.mse[mse_id],
            on_step=True,
            on_epoch=True,
        )

        return self.loss(input=autoenc_output, target=batch)

    def training_step(self, batch, batch_idx: int):
        # why does the train dataloader return list with a stacked tensor
        # and the valid dataloader returns the tensor itself???
        assert isinstance(batch, list)
        return self.base_step(batch[0], batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    # TODO: missing: checkpointing (?) (see lightning.pytorch.callbacks import ModelCheckpoint) # noqa: E501
    # TODO: move this comment to the training node
