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
import torchdrift.utils

__all__ = []

_common_doc = """

    Args:
        x: input image, tensor in 0..1 range

        severity: integer severity from 1 to 5

    The intensity adapts to the image size to interpolate between the parameters
    they set by Hendrycks and Dieterich for CIFAR and ImageNet.
"""


def _export(fn):
    __all__.append(fn.__name__)

    fn.__doc__ = fn.__doc__ + _common_doc
    return fn


def interpolate_severity(img, cifar, imagenet, severity):
    torchdrift.utils.check(
        severity >= 1 and severity <= 5, "severity needs to be between 1 and 5"
    )
    length = (img.size(-1) * img.size(-2)) ** 0.5
    alpha = max(min((length - 32) / (224 - 32), 1), 0)
    res = (1 - alpha) * cifar[severity - 1] + alpha * imagenet[severity - 1]
    if isinstance(cifar[0], int):
        res = int(res)
    return res


@_export
def gaussian_noise(x: Tensor, severity: int = 1) -> Tensor:
    """Applys gaussian noise."""
    cifar_std = [0.04, 0.06, 0.08, 0.09, 0.10]
    imagenet_std = [0.08, 0.12, 0.18, 0.26, 0.38]
    std = interpolate_severity(x, cifar_std, imagenet_std, severity)
    return (x + torch.randn_like(x) * std).clamp(min=0.0, max=1.0)


@_export
def shot_noise(x: Tensor, severity: int = 1) -> Tensor:
    """Applys shot noise."""
    imagenet_rate_mult = [60, 25, 12, 5, 3]
    cifar_rate_mult = [500, 250, 100, 75, 50]
    rate_mult = interpolate_severity(x, cifar_rate_mult, imagenet_rate_mult, severity)
    # very likely the clamp is not needed, but numerical stability...
    return (torch.poisson(x * rate_mult) / rate_mult).clamp(min=0.0, max=1.0)


@_export
def impulse_noise(x: Tensor, severity: int = 1) -> Tensor:
    """Applies impulse noise"""
    imagenet_amount = [0.03, 0.06, 0.09, 0.17, 0.27]
    cifar_amount = [0.01, 0.02, 0.03, 0.05, 0.07]
    amount = interpolate_severity(x, cifar_amount, imagenet_amount, severity)
    salt = torch.bernoulli(torch.full_like(x, amount / 2))
    pepper = torch.bernoulli(torch.full_like(x, 1 - amount / 2))
    return torch.max(torch.min(x, pepper), salt)


@_export
def speckle_noise(x: Tensor, severity: int = 1) -> Tensor:
    """Applies speckle noise"""
    imagenet_c = [0.15, 0.2, 0.35, 0.45, 0.6]
    cifar_c = [0.06, 0.1, 0.12, 0.16, 0.2]
    c = interpolate_severity(x, cifar_c, imagenet_c, severity)
    return (x + x * c * torch.randn_like(x)).clamp(min=0.0, max=1.0)


@_export
def gaussian_blur(x: Tensor, severity: int = 1) -> Tensor:
    """Applies gaussian blur"""
    imagenet_sigma = [1, 2, 3, 4, 6]
    cifar_sigma = [0.4, 0.6, 0.7, 0.8, 1]
    sigma = interpolate_severity(x, cifar_sigma, imagenet_sigma, severity)

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
