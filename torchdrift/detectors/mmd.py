from typing import Optional

import torch

from . import DriftDetector

def kernel_mmd(x, y, n_perm=1000):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
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


class KernelMMDDriftDetector(DriftDetector):
    def predict_shift_from_features(self, base_outputs: torch.Tensor, outputs: torch.Tensor, individual_samples: bool = False):
        assert (
            not individual_samples
        ), "Individual samples not supported by MMD detector"
        ood_score = kernel_mmd(
            outputs, self.base_outputs, n_perm=None
        )  # we have higher == more abnormal
        return ood_score
