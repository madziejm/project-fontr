import torch

from fontr.fontr.autoencoder import Autoencoder


def test_autoencoder_forward():
    autoenc = Autoencoder(lr=0.01)
    example_input_array = torch.rand(
        1, 1, 96, 96, dtype=torch.float32, requires_grad=False
    )  # used for serialization by tracing with torchscript
    reconstructed_x = autoenc(example_input_array)
    assert reconstructed_x.shape == example_input_array.shape
