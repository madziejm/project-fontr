import tempfile

import torch

from fontr.fontr.autoencoder import Autoencoder
from fontr.fontr.classifier import Classifier


def test_classsifier_forward():
    lr = 0.01
    _input = torch.rand(1, 1, 96, 96)  # grayscale thus single channel
    classifier = Classifier(lr)
    classifier(_input)


def test_classifier_from_autoencoder():
    """Tests whether Classifier can be initialized with a serialized Autoencoder"""
    autoenc = Autoencoder(lr=0.42)

    with tempfile.TemporaryFile("w+b") as f:
        autoenc.eval()
        with torch.no_grad():
            torch.save(autoenc, f)
        del autoenc
        f.seek(0)
        autoenc = torch.load(f)

    Classifier(0.42, autoenc)
