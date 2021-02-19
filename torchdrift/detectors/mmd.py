from typing import Optional

import torch

from . import Detector

def kernel_mmd(x, y, n_perm=1000):
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
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / dists[:100, :100].median() ** 2) * dists ** 2)
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
    p_val = count / n_perm
    return mmd, p_val


class KernelMMDDriftDetector(Detector):
    """Drift detector based on the kernel Maximum Mean Discrepancy (MMD) test.

This is modelled after the MMD drift detection in
S. Rabanser et al: *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* (NeurIPS), 2019.

Note that our heuristic choice of the kernel bandwith is more closely aligned with that of the original MMD paper and code than S. Rabanser's.
    """
    
    def __init__(self, *, return_p_value=False, n_perm: int = 1000):
        super().__init__(return_p_value=return_p_value)
        self.n_perm = n_perm

    def predict_shift_from_features(self, base_outputs: torch.Tensor, outputs: torch.Tensor, compute_score: bool, compute_p_value: bool, individual_samples: bool = False):
        assert (
            not individual_samples
        ), "Individual samples not supported by MMD detector"
        if not compute_p_value:
            ood_score = kernel_mmd(
                outputs, base_outputs, n_perm=None
            )
            p_value = None
        else:
            ood_score, p_value = kernel_mmd(
                outputs, base_outputs, n_perm=self.n_perm
            )
        return ood_score, p_value
