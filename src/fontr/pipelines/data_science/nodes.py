"""todo nuke this file
"""

import logging
from multiprocessing import cpu_count

import pytorch_lightning as pl
import torch
import torchvision
import tqdm
from nodes.data_science import Autoencoder
from PIL import Image

from pytorch_lightning.accuracy import (
    Accuracy,
)  # use accuracy and pass multiclass param # todo
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.jit import ScriptModule
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassPrecision, MulticlassRecall

from fontr.datasets import KedroPytorchImageDataset
from fontr.pipelines.data_science.transformers import AddGaussianNoise, ResizeImage


def get_dataloader(dataset, batch_size, num_workers=0):
    # todo
    raise torch.utils.data.DataLoader(
        dataset, batch_size, num_workers=num_workers, shuffle=True
    )


def train_pytorch_autoencoder(
    train_dataset: KedroPytorchImageDataset,
    val_dataset: KedroPytorchImageDataset,
    label2index: dict,
    parameters: dict,
):
    """Trains the autoencoder.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained autoencoder.
    """
    pl.seed_everything(parameters["random_state_seed"])

    autoencoder = Autoencoder(lr=parameters["lr"])

    trainer = pl.Trainer(
        max_epochs=parameters["maxnepochs"],
        logger=True,
        max_steps=parameters.get("max_steps", -1),
        accelerator="auto",
        callbacks=[TQDMProgressBar()],
    )
    target_transform = transforms.Compose([
        #transforms.ToTensor(),
            AddGaussianNoise(0., 3.), 
            transforms.GaussianBlur(kernel_size=3, sigma=(2.5, 3.5)),
            transforms.RandomAffine(degrees=(-4, 4)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
            transforms.Grayscale(), 
            ResizeImage(96), 
            transforms.RandomCrop((96, 96))
        ])
    trainer.fit(
        autoencoder,
        train_dataloaders=[
            get_dataloader(
                train_dataset.with_transforms(
                    target_transform=target_transform,  # todo get this transform
                ),
                parameters["batch_size"],
                num_workers=parameters.get("num_workers", cpu_count()),
            )
        ],
        val_dataloaders=[
            get_dataloader(
                val_dataset.with_transforms(
                    target_transform=target_transform  # todo get this transform
                ),
                parameters["batch_size"],
                # train=False, TODO:
            )
        ],
    )

    return autoencoder.to_torchscript()


# todo add train_pytorch_classifier function

# todo add evaluate_autoencoder


@torch.no_grad()
def evaluate_classifier(
    classifier: ScriptModule,
    test_dataset: KedroPytorchImageDataset,
    label2index: dict,
    parameters: dict,
):
    """Evaluate classifier on test dataset

    Args:
        classifier (ScriptModule): Trained classifier
        test_dataset (KedroPytorchImageDataset): test dataset
        label2index (dict): labels
        parameters (dict): pipeline parameters
    """
    data_loader = get_dataloader(
        test_dataset.with_transforms(
            # target_transform=todo
        ),
        parameters["batch_size"],
        # train=False, TODO:
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(label2index)

    precision, recall, accuracy = (
        MulticlassPrecision(num_classes),
        MulticlassRecall(num_classes),
        Accuracy(task="multiclass", num_classes=num_classes),
    )

    for batch in tqdm.tqdm(iter(data_loader), "Test set evaluation", len(test_dataset)):
        x, y = batch
        preds = classifier(x.to(device).cpu())
        precision.update(preds, y)
        recall.update(preds, y)
        accuracy.update(preds, y)

    logging.info(f"Precision{precision.compute():0.3f}")
    logging.info(f"Recall{recall.compute():0.3f}")
    logging.info(f"Accuracy{accuracy.compute():0.3f}")


@torch.no_grad()
def predict(
    classifier: ScriptModule,
    # test_dataset: KedroPytorchImageDataset,
    file_to_predict: str,
    label2index: dict,
    parameters: dict,
):
    img = Image.open(file_to_predict).convert("RGB")  # resize?
    x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    preds = classifier(x).squeeze()

    print(preds)
