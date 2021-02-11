#!/usr/bin/env python
# coding: utf-8

# TODO(tv): Check if we need to distinguish cifar/imagenet and how that would look like

# The functions below aim to provide equivalents for the functions used in
#    Dan Hendrycks and Thomas Dietterich
#    Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR 2019).
# and provided by their authors in https://github.com/hendrycks/robustness/ to create CIFAR10-C

# We seek to provide PyTorch functions operating on the typical batch x c x h x w tensors.
# Entries must be between 0 and 1 (so *before* normalizaiton).
# The functions should work on both CPU and GPU tensors.

# As we do not use the same random number generators, you cannot expect to get the exact same result.
# Also we may have differences in implementation (e.g. different choices of cutoffs in smoothing kernels),
# these are expected.
# Until this version becomes more popular, you probably want to use the original code or the pre-processed
# datasets for your publication.

# Done: Gaussian Noise, Shot Noise, Impulse Noise, Speckle Noise, Gaussian Blur

# Todo: Defocus Blur, Glass Blur, Motion Blur, Zoom Blur, Snow, Frost, Fog,
#       Brightness, Contrast, Elastic, Pixelate, JPEG, Spatter, Saturate

from torch import Tensor
import torch
import math

# TODO(tv): pick std based on optional arg or size of input


def gaussian_noise(x: Tensor, severity: int = 1) -> Tensor:
    # std = [0.04, 0.06, .08, .09, .10][severity - 1]  # CIFAR-C
    std = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]  # ImageNet-C
    return (x + torch.randn_like(x) * std).clamp(min=0.0, max=1.0)


def shot_noise(x, severity=1):
    rate_mult = [60, 25, 12, 5, 3][severity - 1]  # ImageNet-C
    # rate_mult = [500, 250, 100, 75, 50][severity - 1]  # CIFAR-C
    # very likely the clamp is not needed, but numerical stability...
    return (torch.poisson(x * rate_mult) / rate_mult).clamp(min=0.0, max=1.0)


def impulse_noise(x, severity=1):
    amount = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]  # ImageNet-C
    # amount = [.01, .02, .03, .05, .07][severity - 1]  # CIFAR-C
    salt = torch.bernoulli(torch.full_like(x, amount / 2))
    pepper = torch.bernoulli(torch.full_like(x, 1 - amount / 2))
    return torch.max(torch.min(x, pepper), salt)


def speckle_noise(x, severity=1):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]  # Imagenet-C
    # c = [.06, .1, .12, .16, .2][severity - 1]  # CIFAR-C
    return (x + x * c * torch.randn_like(x)).clamp(min=0.0, max=1.0)


def gaussian_blur(x, severity=1):
    sigma = [1, 2, 3, 4, 6][severity - 1]  # Imagenet-C
    # sigma = [.4, .6, 0.7, .8, 1][severity - 1]  # CIFAR-C

    n, channels, h, w = x.shape
    width = int(math.ceil(sigma * 5))
    width = width + (width + 1) % 2  # we want an odd size

    distance = torch.arange(
        -(width // 2), width // 2 + 1, dtype=torch.float, device=x.device
    )
    gaussian = torch.exp(
        -(distance[:, None] ** 2 + distance[None] ** 2) / (2 * sigma ** 2)
    )
    gaussian /= gaussian.sum()
    channels = x.size(1)
    kernel = gaussian[None, None].expand(channels, -1, -1, -1)
    return torch.nn.functional.conv2d(x, kernel, padding=width // 2, groups=channels)
