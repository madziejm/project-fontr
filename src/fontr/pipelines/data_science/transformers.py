import torch
import torchvision.transforms as transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ResizeImage(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, tensor):
        return transforms.Resize(
            (self.height, int(tensor.size()[2] * (self.height / tensor.size()[1])))
        )(tensor)
