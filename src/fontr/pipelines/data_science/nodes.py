import logging
from multiprocessing import cpu_count

import wandb
import pytorch_lightning as pl
import torch
import torchvision
import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.jit import ScriptModule
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)

from fontr.datasets import KedroPytorchImageDataset
from fontr.fontr.autoencoder import Autoencoder
from fontr.fontr.classifier import Classifier
from fontr.fontr.logger import TorchLogger
from fontr.pipelines.data_science.transformers import AddGaussianNoise, ResizeImage
from torchvision import transforms

# from PIL import Image


def get_dataloader(dataset, batch_size, num_workers=None, shuffle=True):
    num_workers = cpu_count() if num_workers is None else num_workers
    return torch.utils.data.DataLoader(
        dataset, batch_size, num_workers=num_workers, shuffle=shuffle
    )


def train_autoencoder(
    train_dataset: KedroPytorchImageDataset,
    val_dataset: KedroPytorchImageDataset,
    parameters: dict,
) -> Autoencoder:
    """Autoencoder training loop.

    Args:
        train_dataset (KedroPytorchImageDataset): Training images.
        val_dataset (KedroPytorchImageDataset): Validation images.
        parameters (dict): Training configuration.

    Returns:
        Autoencoder: Trained autoencoder.
    """

    autoencoder = Autoencoder(lr=parameters["lr"])
    wandb_logger = TorchLogger().getLogger()

    trainer = pl.Trainer(
        max_epochs=parameters["maxnepochs"],
        logger=wandb_logger,
        max_steps=parameters.get("max_steps", -1),
        accelerator="auto",
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=1,
    )

    # TODO add transforms here!!!
    transform = torch.nn.Sequential(
        torchvision.transforms.Grayscale(),
        # TODO welp, actually the images in the paper are 105x105
        # consider changing architecture accordingly (from 96x96 inputs)
        torchvision.transforms.Resize((96, 96)),
    )
    transform_copy = transforms.Compose([
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0.0, 3.0),
        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2.5, 3.5)),
        torchvision.transforms.RandomAffine(degrees=(-4, 4)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
        torchvision.transforms.Grayscale(),
        ResizeImage(96),
        torchvision.transforms.RandomCrop((96, 96)),
    ])
    trainer.fit(
        autoencoder,
        train_dataloaders=[
            get_dataloader(
                train_dataset.with_transforms(
                    transform=transform, copy_transform=transform_copy
                ),
                parameters["batch_size"],
                num_workers=parameters.get("num_workers"),
            )
        ],
        val_dataloaders=[
            get_dataloader(
                val_dataset.with_transforms(transform=transform),
                parameters["batch_size"],
                num_workers=parameters.get("num_workers"),
                shuffle=False,
            )
        ],
    )

    return autoencoder


def train_classifier(
    train_dataset: KedroPytorchImageDataset,
    val_dataset: KedroPytorchImageDataset,
    label2idx: dict,
    parameters: dict,
    autoencoder: Autoencoder,
) -> Classifier:
    """Font classifier training loop.

    Args:
        train_dataset (KedroPytorchImageDataset): Training images.
        val_dataset (KedroPytorchImageDataset): Validation images.
        label2idx (dict): Mapping from label name to label index.
        parameters (dict): Training parameters

    Returns:
        Classifier: Trained classifier.
    """

    classifier = Classifier(
        lr=parameters["lr"], autoencoder=autoencoder, nclasses=len(label2idx)
    )
    wandb_logger = TorchLogger().getLogger()

    trainer = pl.Trainer(
        max_epochs=parameters["maxnepochs"],
        logger=wandb_logger,
        max_steps=parameters.get("max_steps", -1),
        accelerator="auto",
        callbacks=[TQDMProgressBar()],
        log_every_n_steps=1,
    )

    # TODO add transforms here!!!
    transform = torch.nn.Sequential(
        torchvision.transforms.Grayscale(),
        # TODO welp, actually the images in the paper are 105x105
        # consider changing architecture accordingly (from 96x96 inputs)
        torchvision.transforms.Resize((96, 96)),
    )

    transform_copy = transforms.Compose([
        torchvision.transforms.ToTensor(),
        AddGaussianNoise(0.0, 3.0),
        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(2.5, 3.5)),
        torchvision.transforms.RandomAffine(degrees=(-4, 4)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=1.0),
        torchvision.transforms.Grayscale(),
        ResizeImage(96),
        torchvision.transforms.RandomCrop((96, 96)),
    ])

    trainer.fit(
        classifier,
        train_dataloaders=[
            get_dataloader(
                train_dataset.with_transforms(
                    transform=transform, copy_transform=transform_copy
                ),
                parameters["batch_size"],
                num_workers=parameters.get("num_workers"),
            )
        ],
        val_dataloaders=[
            get_dataloader(
                val_dataset.with_transforms(transform=transform),
                parameters["batch_size"],
                num_workers=parameters.get("num_workers"),
                shuffle=False,
            )
        ],
    )

    return classifier


def serialize_model_to_torch_jit(
    model: pl.LightningModule, torch_jit_serialization_method: str
) -> ScriptModule:
    """Serialize pl.LightningModule object to TorchScript JIT format

    Args:
        model (pl.LightningModule): Model to be serialized
        torch_jit_serialization_method (str): 'trace' or 'script'

    Returns:
        ScriptModule: Serialized model
    """
    return model.to_torchscript(method=torch_jit_serialization_method)  # type: ignore


@torch.no_grad()
def evaluate_autoencoder(
    autoencoder: ScriptModule,
    test_dataset: KedroPytorchImageDataset,
    parameters: dict,
):
    """Evaluate autoencoder on test dataset. TODO implement this

    Args:
        autoencoder (ScriptModule): Autoencoder.
        test_dataset (KedroPytorchImageDataset): Test set images.
        parameters (dict): Evaluation parameters

    Raises:
        NotImplementedError: Raised on every invocation.
    """
    # TODO: Implement storing MSE scores using W&B.
    raise NotImplementedError


@torch.no_grad()
def evaluate_classifier(
    classifier: ScriptModule,
    test_dataset: KedroPytorchImageDataset,
    label2idx: dict,
    parameters: dict,
):
    """Evaluate classifier on test dataset

    Args:
        classifier (ScriptModule): Trained classifier
        test_dataset (KedroPytorchImageDataset): test dataset
        label2idx (dict): labels
        parameters (dict): pipeline parameters
    """
    data_loader = get_dataloader(
        test_dataset.with_transforms(
            # transform=TODO
        ),
        parameters["batch_size"],
        num_workers=parameters.get("num_workers"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = len(label2idx)

    precision, recall, accuracy = (
        MulticlassPrecision(num_classes),
        MulticlassRecall(num_classes),
        MulticlassAccuracy(num_classes),
    )

    for batch in tqdm.tqdm(iter(data_loader), "Test set evaluation", len(test_dataset)):
        x, y = batch
        preds = classifier(x.to(device).cpu())
        precision.update(preds, y)
        recall.update(preds, y)
        accuracy.update(preds, y)

    wandb_logger = TorchLogger().getLogger()
    columns = ["Precision", "Recall", "Accuracy"]
    data = [
        [
            f"{precision.compute():0.3f}",
            f"{recall.compute():0.3f}",
            f"{accuracy.compute():0.3f}",
        ]
    ]
    wandb_logger.log_text("Classifier evaluation", columns=columns, data=data)


@torch.no_grad()
def predict(
    classifier: ScriptModule,
    # test_dataset: KedroPytorchImageDataset,
    file_to_predict: str,
    label2idx: dict,
    parameters: dict,
):
    """TODO do we actually need this inference node? is this for evaluation?"""
    raise NotImplementedError
    # img = Image.open(file_to_predict).convert("RGB")  # resize?
    # x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    # preds = classifier(x).squeeze()
