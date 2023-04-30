from typing import Any

import pytorch_lightning as pl


class ClassificationModel(pl.LightningModule):
    def __init__(self) -> None:  # todo add autoencoder pickle here
        super.__init__()
        self.lr = 0.0  # todo add param

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)  # todo

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)  # todo

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)  # todo
