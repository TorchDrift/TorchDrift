from typing import Optional

import torch

from . import DriftDetector

# TODO: ks p-value computation...

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
    return ks_scores.max()

class KSDriftDetector(DriftDetector):
    def predict_shift_from_features(self, base_outputs: torch.Tensor, outputs: torch.Tensor, individual_samples: bool = False):
        assert (
            not individual_samples
        ), "Individual samples not supported by KS detector"
        ood_score = ks_two_sample_multi_dim(outputs, self.base_outputs)
        return ood_score
