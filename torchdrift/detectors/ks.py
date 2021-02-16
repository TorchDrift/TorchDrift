from typing import Optional

import torch
import numpy

from . import DriftDetector

try:
    import numba
    njit = numba.jit(nopython=True, fastmath=True)
except ImportError:
    njit = lambda x: x


# Numerically stable p-Value computation, see
# T. Viehmann: Numerically more stable computation of the p-values for the
#              two-sample Kolmogorov-Smirnov test
# https://arxiv.org/abs/2102.xxxxx

@njit
def ks_p_value(n, m, d):
    size = int(2*m*d+2)
    lastrow, row = numpy.zeros((2, size), dtype=numpy.float64)
    last_start_j = 0
    for i in range(n + 1):
        start_j = max(int(m * (i/n + d)) + 1-size, 0)
        lastrow, row = row, lastrow
        val = 0.0
        for jj in range(size):
            j = jj + start_j
            dist = i/n - j/m
            if dist > d or dist < -d:
                val = 1.0
            elif i == 0 or j == 0:
                val = 0.0
            elif jj + start_j - last_start_j >= size:
                val = (i + val * j) / (i + j)
            else:
                val = (lastrow[jj + start_j - last_start_j] * i + val * j) / (i + j)
            row[jj] = val
        jjmax = min(size, m + 1 - start_j)
        last_start_j = start_j
    return row[m - start_j]

def ks_two_sample_multi_dim(x, y):
    """
    As in failing loudly, we perform the KS test per dimension.
    """
    n_x, n_features = x.shape
    n_y, n_features_y = y.shape
    assert n_features == n_features_y
    
    joint_sorted = torch.argsort(torch.cat([x, y], dim=0), dim=0)
    sign = (joint_sorted < n_x).to(dtype=torch.float) * (1 /(n_x) + 1/(n_y)) - (1/(n_y))
    ks_scores = sign.cumsum(0).abs().max(0).values

    # Like failing loudly suggests to return the minimum p-Value under the
    # label Bonferroni correction, this would correspond to the maximum score
    # see the p-value computation below...
    return ks_scores.max()

class KSDriftDetector(DriftDetector):
    def predict_shift_from_features(self, base_outputs: torch.Tensor, outputs: torch.Tensor, compute_score: bool, compute_p_value: bool, individual_samples: bool = False):
        assert (
            not individual_samples
        ), "Individual samples not supported by MMD detector"
        ood_score = ks_two_sample_multi_dim(outputs, self.base_outputs)
        if compute_p_value:
            nx, n_features = base_outputs.shape
            ny, _ = outputs.shape
            # multiply by n_features for Bonferroni correction.
            p_value = min(1.0, ks_p_value(nx, ny, ood_score.item()) * n_features)
        else:
            p_value = None
        return ood_score, p_value
