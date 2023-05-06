import torch

from fontr.fontr.autoencoder import Autoencoder


def test_autoencoder_forward():
    lr = 0.01
    _input = torch.rand(3, 115, 115)
    autoenc = Autoencoder(lr)
    autoenc(_input)
