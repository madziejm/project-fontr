import torch

from fontr.fontr.classifier import Classifier


def test_classsifier_forward():
    lr = 0.01
    _input = torch.rand(1, 3, 115, 115)
    classifier = Classifier(lr)
    classifier(_input)
