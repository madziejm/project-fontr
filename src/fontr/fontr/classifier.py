from typing import Any, Optional

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import MulticlassAccuracy

from fontr.fontr.autoencoder import Autoencoder


class Classifier(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        autoencoder: Optional[Autoencoder] = None,
        nclasses: int = 2383,
        dropout=0.1,
    ) -> None:
        """Construct a font classifier object as described in "DeepFont: Identify Your Font from An Image" paper # noqa: E501

        Args:
            lr (float): learning rate.
            autoencoder (Optional[Autoencoder], optional): Autoencoder to initialize lower layers with. Defaults to None. # noqa: E501
            nclasses (int): number of classes. Defaults to 2383.
        """
        super().__init__()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.nclasses = nclasses

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        if autoencoder:
            assert len(self.encoder) == len(autoencoder.encoder)

            for layer_to_be_initialized, layer_trained in zip(
                self.encoder, autoencoder.encoder
            ):
                assert type(layer_to_be_initialized) is type(layer_trained)
                layer_to_be_initialized.load_state_dict(layer_trained.state_dict())
                if isinstance(layer_to_be_initialized, nn.MaxPool2d):
                    layer_to_be_initialized.return_indices = False

            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier_suffix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.Flatten(start_dim=1),
            nn.LeakyReLU(0.1),
            nn.Linear(36864, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, self.nclasses),
        )

        self.accuracy = nn.ModuleDict(
            {
                "train_accuracy": MulticlassAccuracy(num_classes=self.nclasses),
                "test_accuracy": MulticlassAccuracy(num_classes=self.nclasses),
                "val_accuracy": MulticlassAccuracy(num_classes=self.nclasses),
            }
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier_suffix(x)
        return x

    # def on_train_start(self) -> None:
    #     # actually logging strings is not supported in pl
    #     # TODO adjust or remove
    #     super().on_train_start()
    #     assert self.logger is not None
    #     self.logger.log("encoder", str(self.encoder))
    #     self.logger.log("classifier_suffix", str(self.classifier_suffix))

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # TODO: 15 patch-averaging here?
        return nn.functional.softmax(self.forward(batch), dim=1)

    def base_step(self, batch, batch_idx: int, step_name: str):
        # TODO somehow we do not get labels heer
        # and this fails
        # it should not be the case as the DataSet seems to be okay
        # is it the DataLoader to blame?
        x, y = batch

        logits = self.forward(x)
        accuracy_id: str = step_name + "_accuracy"
        self.accuracy[accuracy_id].forward(logits, y)
        self.log(
            accuracy_id,
            self.accuracy[accuracy_id],
            on_step=True,
            on_epoch=True,
        )
        return self.criterion(input=logits, target=y)

    def training_step(self, batch, batch_idx):
        # TODO implement lr strategy "We start with the learning rate at 0.01, and follow a common heuristic to manually divide the learning rate by 10 when the validation error rate stops decreasing with the current rate" # noqa: E501
        # TODO document that batch_idx is not used (it comes from Lightning)
        return self.base_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005
        )  # TODO parametrize momentum and weight decay here
