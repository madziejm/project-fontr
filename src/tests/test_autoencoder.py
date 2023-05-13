import torch

from fontr.fontr.autoencoder import Autoencoder


def test_autoencoder_forward():
    lr = 0.01
    _input = torch.rand(
        1, 1, 96, 96, dtype=torch.float32
    )  # grayscale thus single channel
    autoenc = Autoencoder(lr)
    autoenc(_input)
