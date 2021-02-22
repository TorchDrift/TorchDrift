from typing import Optional

import torch
import numpy

from . import Detector

try:
    import numba

    njit = numba.jit(nopython=True, fastmath=True)
except ImportError:  # pragma: no cover
    njit = lambda x: x


# Numerically stable p-Value computation, see
# T. Viehmann: Numerically more stable computation of the p-values for the
#              two-sample Kolmogorov-Smirnov test
# https://arxiv.org/abs/2102.08037


@njit
def ks_p_value(n: int, m: int, d: float) -> float:
    """Computes the p-value for the two-sided two-sample KS test from the D-statistic.

    This uses the stable recursion from T. Viehmann: Numerically more stable computation of the p-values for the two-sample Kolmogorov-Smirnov test.
    """
    size = int(2 * m * d + 2)
    lastrow, row = numpy.zeros((2, size), dtype=numpy.float64)
    last_start_j = 0
    for i in range(n + 1):
        start_j = max(int(m * (i / n + d)) + 1 - size, 0)
        lastrow, row = row, lastrow
        val = 0.0
        for jj in range(size):
            j = jj + start_j
            dist = i / n - j / m
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


def ks_two_sample_multi_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the two-sample two-sided Kolmorogov-Smirnov statistic.

    The inputs `x` and `y` are expected to be 2-dimensional tensors,
    the first dimensions being batch (potentially different) and the second
    features (necessarily the same).

    We return a one-dimensional tensor of KS scores D, one per dimension.
    """
    n_x, n_features = x.shape
    n_y, n_features_y = y.shape
    assert n_features == n_features_y

    joint_sorted = torch.argsort(torch.cat([x, y], dim=0), dim=0)
    sign = (joint_sorted < n_x).to(dtype=torch.float) * (1 / (n_x) + 1 / (n_y)) - (
        1 / (n_y)
    )
    ks_scores = sign.cumsum(0).abs().max(0).values
    return ks_scores


class KSDriftDetector(Detector):
    """Drift detector based on (multiple) Kolmogorov-Smirnov tests.

    This detector uses the Kolmogorov-Smirnov test on the marginals of the features
    for each feature.

    For scores, it returns the maximum score. p-values are computed with the
    Bonferroni correction of multiplying the p-value of the maximum score by
    the number of features/tests.

    This is modelled after the KS drift detection in
    S. Rabanser et al: *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* (NeurIPS), 2019.
    """

    def predict_shift_from_features(
        self,
        base_outputs: torch.Tensor,
        outputs: torch.Tensor,
        compute_score: bool,
        compute_p_value: bool,
        individual_samples: bool = False,
    ):
        assert (
            not individual_samples
        ), "Individual samples not supported by MMD detector"
        ood_score = ks_two_sample_multi_dim(outputs, self.base_outputs)
        # Like failing loudly suggests to return the minimum p-value under the
        # label Bonferroni correction, this would correspond to the maximum score
        # see the p-value computation below...
        ood_score = ood_score.max()

        if compute_p_value:
            nx, n_features = base_outputs.shape
            ny, _ = outputs.shape
            # multiply by n_features for Bonferroni correction.
            p_value = min(1.0, ks_p_value(nx, ny, ood_score.item()) * n_features)
        else:
            p_value = None
        return ood_score, p_value
