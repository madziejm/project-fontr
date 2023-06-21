from typing import Any, Optional

import pytorch_lightning as pl
from torch import nn, max
from torch.optim import Adam
from torchmetrics.classification import Accuracy

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
        self.top_k = 10

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

        self.accuracy = {
            "train_accuracy": Accuracy(task="multiclass", num_classes=self.nclasses),
            "test_accuracy": Accuracy(task="multiclass", num_classes=self.nclasses),
            "val_accuracy": Accuracy(task="multiclass", num_classes=self.nclasses),
        }

        self.top_k_accuracy = {
            f"train_top_{self.top_k}_accuracy": Accuracy(
                task="multiclass", num_classes=self.nclasses, top_k=self.top_k
            ),
            f"test_top_{self.top_k}_accuracy": Accuracy(
                task="multiclass", num_classes=self.nclasses, top_k=self.top_k
            ),
            f"val_top_{self.top_k}_accuracy": Accuracy(
                task="multiclass", num_classes=self.nclasses, top_k=self.top_k
            ),
        }
        self.save_hyperparameters()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier_suffix(x)
        return x

    def predict_step(self, x: Any, batch_idx: int, dataloader_idx: int = 0):
        output = self.forward(x)
        output = max(output, 0).values
        softmax = nn.functional.softmax(output)
        return softmax

    def base_step(self, batch, batch_idx: int, step_name: str):
        x, y = batch
        b, p, c, h, w = x.shape
        flatten_x = x.view(b * p, c, h, w)
        output = self.forward(flatten_x)

        output = output.view(b, p, -1)
        # Getting max from every set of patches
        output = max(output, 1).values
        softmax = nn.functional.softmax(output, dim=1)

        accuracy_id = step_name + "_accuracy"
        self.accuracy[accuracy_id].forward(softmax, y)  # type: ignore
        self.log(
            accuracy_id,
            self.accuracy[accuracy_id],  # type: ignore
            on_step=True,
            on_epoch=True,
        )
        top_k_accuracy_id = step_name + f"_top_{self.top_k}_accuracy"
        self.top_k_accuracy[top_k_accuracy_id].forward(softmax, y)  # type: ignore
        self.log(
            top_k_accuracy_id,
            self.top_k_accuracy[top_k_accuracy_id],  # type: ignore
            on_step=True,
            on_epoch=True,
        )

        cross_entropy = self.criterion(input=output, target=y)
        self.log(
            step_name + "_cross_entropy",
            cross_entropy,
            on_step=True,
            on_epoch=True,
        )
        return cross_entropy

    def training_step(self, batch, batch_idx):
        # TODO implement lr strategy "We start with the learning rate at 0.01, and follow a common heuristic to manually divide the learning rate by 10 when the validation error rate stops decreasing with the current rate" # noqa: E501
        # TODO document that batch_idx is not used (it comes from Lightning)
        return self.base_step(batch[0], batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        return self.base_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=0.0005)
