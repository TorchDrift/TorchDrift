from typing import Optional

import torch

from . import Detector
import torchdrift.utils


class Kernel:
    """Base class for kernels

    Unless otherwise noted, all kernels implementing lengthscale detection
    use the median of pairwise distances as the lengthscale."""

    pass


class GaussianKernel(Kernel):
    r"""Unnormalized gaussian kernel

    .. math::
        k(|x-y|) = \exp(-|x-y|^2/(2\ell^2))

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def __call__(self, dists):
        # note that lengthscale should be squared in the RBF to match the Gretton et al heuristic
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.exp((-0.5 / lengthscale ** 2) * dists ** 2)


class ExpKernel(Kernel):
    r"""Unnormalized exponential kernel

    .. math::
        k(|x-y|) = \exp(-|x-y|/\ell)

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def __call__(self, dists):
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.exp((-1 / lengthscale) * dists)


class RationalQuadraticKernel(Kernel):
    r"""Unnormalized rational quadratic kernel

    .. math::
        k(|x-y|) = (1+|x-y|^2/(2 \alpha \ell^2))^{-\alpha}

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.lengthscale = lengthscale

    def __call__(self, dists):
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.pow(
            1 + (1 / (2 * self.alpha * lengthscale ** 2)) * dists ** 2, -self.alpha
        )


def kernel_mmd(x, y, n_perm=1000, kernel=GaussianKernel()):
    """Implements the kernel MMD two-sample test.

    It is modelled after the kernel MMD paper and code:
    A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    The arguments `x` and `y` should be two-dimensional tensors.
    The first is the batch dimension (which may differ), the second
    the features (which must be the same on both `x` and `y`).

    `n_perm` is number of bootstrap permutations to get p-value, pass `None` to not get p-value.
    """

    n, d = x.shape
    m, d2 = y.shape
    torchdrift.utils.check(d == d2, "feature dimension mismatch")
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    k = kernel(dists)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = (
        k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    )
    if n_perm is None:
        return mmd
    mmd_0s = []
    count = 0
    for i in range(n_perm):
        # this isn't efficient, it would be lovely to generate a cuda kernel or C++ for loop and do the
        # permutation on the fly...
        pi = torch.randperm(n + m, device=x.device)
        k = k[pi][:, pi]
        k_x = k[:n, :n]
        k_y = k[n:, n:]
        k_xy = k[:n, n:]
        # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
        mmd_0 = (
            k_x.sum() / (n * (n - 1))
            + k_y.sum() / (m * (m - 1))
            - 2 * k_xy.sum() / (n * m)
        )
        mmd_0s.append(mmd_0)
        count = count + (mmd_0 > mmd)
    # pyplot.hist(torch.stack(mmd_0s, dim=0).tolist(), bins=50)
    # true_divide: torch 1.6 compat replace with "/" after October 2021
    p_val = torch.true_divide(count, n_perm)

    return mmd, p_val


class KernelMMDDriftDetector(Detector):
    """Drift detector based on the kernel Maximum Mean Discrepancy (MMD) test.

    This is modelled after the MMD drift detection in
    S. Rabanser et al: *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* (NeurIPS), 2019.

    Note that our heuristic choice of the kernel bandwith is more closely aligned with that of the original MMD paper and code than S. Rabanser's.

    The default kernel is the unnormalized Gaussian (or Squared Exponential) kernel.
    """

    def __init__(
        self, *, return_p_value=False, n_perm: int = 1000, kernel=GaussianKernel()
    ):
        super().__init__(return_p_value=return_p_value)
        self.n_perm = n_perm
        self.kernel = kernel
        self.n_test = None
        self.scores = None

    def fit(self, x: torch.Tensor, n_test=None):
        """Record a sample as the reference distribution

        Args:
            x: The reference data
            n_test: If an int is specified, the last n_test datapoints
                will not be considered part of the reference data. Instead,
                bootstrappin using permutations will be used to determine
                the distribution under the null hypothesis at fit time.
                Future testing must then always be done with n_test elements
                to get p-values."""
        x = x.detach()
        if n_test is None:
            self.base_outputs = x
        else:
            torchdrift.utils.check(
                0 < n_test < x.size(0),
                "n_test must be strictly between 0 and the number of samples",
            )
            self.n_test = n_test
            self.base_outputs = x[:-n_test]

            n_ref = x.size(0) - n_test

            scores = []
            for i in range(self.n_perm):
                slicing = torch.randperm(x.size(0))
                scores.append(
                    kernel_mmd(
                        x[slicing[:-n_test]],
                        x[slicing[-n_test:]],
                        n_perm=None,
                        kernel=self.kernel,
                    )
                )
            scores = torch.stack(scores)

            # limited smallish sample sizes, the MMD appears to exhibit a nonzero offset
            # which vanishes in the limit we adapt. After correcting this, the gamma distribution
            # approximation suggested by Gretton et al seems very good.
            self.dist_min = scores.min().double()
            mean = scores.mean() - self.dist_min
            var = scores.var().double()
            self.dist_alpha = mean ** 2 / var
            self.dist_beta = mean / var
            self.scores = scores

    def predict_shift_from_features(
        self,
        base_outputs: torch.Tensor,
        outputs: torch.Tensor,
        compute_score: bool,
        compute_p_value: bool,
        individual_samples: bool = False,
    ):
        torchdrift.utils.check(
            not individual_samples, "Individual samples not supported by MMD detector"
        )
        if not compute_p_value:
            ood_score = kernel_mmd(
                base_outputs, outputs, n_perm=None, kernel=self.kernel
            )
            p_value = None
        elif self.n_test is None:
            ood_score, p_value = kernel_mmd(
                base_outputs, outputs, n_perm=self.n_perm, kernel=self.kernel
            )
        else:
            torchdrift.utils.check(
                self.n_test == outputs.size(0),
                "number of test samples does not match calibrated number",
            )
            ood_score = kernel_mmd(
                base_outputs, outputs, n_perm=None, kernel=self.kernel
            )
            p_value = torch.igammac(
                self.dist_alpha,
                self.dist_beta * (ood_score - self.dist_min).clamp(min=0),
            )  # needs PyTorch >=1.8

        return ood_score, p_value
