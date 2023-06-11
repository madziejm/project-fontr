import torchvision.transforms as transforms
import torch
import random


class Patch(torch.nn.Module):
    """
    Creates patches from images
    """

    def __init__(self, patch_size=10, patch_width=96, patch_height=96):
        self.patch_size = patch_size
        self.patch_width = patch_width
        self.patch_height = patch_height

    def __call__(self, image):
        c, h, w = image.shape

        patches = []

        for _ in range(self.patch_size):
            if w > self.patch_width:
                left = random.randint(0, w - self.patch_width)
                right = left + self.patch_width
            else:
                left = 0
                right = self.patch_width

            if h > self.patch_height:
                up = random.randint(0, h - self.patch_height)
                down = up + self.patch_height
            else:
                up = 0
                down = self.patch_height

            after = image[0, up:down, left:right].unsqueeze(0)
            patches.append(after)
        return torch.stack(patches)


class AddGaussianNoise(torch.nn.Module):
    """
    Adds gaussian noise to a given tensor
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ResizePatches(torch.nn.Module):
    """
    Resizes given patches to defines width and height
    """

    def __init__(self, height=96, width=96):
        self.height = height
        self.width = width

    def __call__(self, patches):
        resized_patches = transforms.Resize((self.width, self.height))(patches)
        return resized_patches
