from typing import Optional

import torch

from . import Detector
import torchdrift.utils

try:
    from ot import emd as ot_emd
except ImportError:  # pragma: no cover

    def ot_emd(x, y, d):
        raise RuntimeError("could not import Python Optimal Transport (ot)")


def wasserstein(x, y, p=2.0, fraction_to_match=1.0, n_perm=1000, return_coupling=False):
    """Wasserstein distance based two-sample test with partial match.
    (see T. Viehmann: Partial Wasserstein and Maximum Mean Discrepancy distances
    for bridging the gap between outlier detection and drift detection,
    https://arxiv.org/abs/2106.01289 )

    Args:
        x: reference distribution sample
        y: test distribution sample
        fraction_to_match: fraction of x probability mass to be matched
        n_perm: number of bootstrap permutations to run to compute p-value (None for not returning a p-value)
        return_coupling: return the coupling

    Note that this currently runs on the CPU and in 64 bit but it will
    move and cast as needed.
    """
    device = x.device
    dtype = x.dtype
    x = x.to(dtype=torch.float64, device="cpu")
    y = y.to(dtype=torch.float64, device="cpu")
    num_x, d = x.shape
    num_y, d2 = y.shape
    torchdrift.utils.check(d == d2, "Shape mismatch in feature dimension")

    dists_p = torch.cdist(x, y, p=p) ** p
    if fraction_to_match < 1.0:
        max_dists_p = dists_p.max()
        dists_p = torch.cat([dists_p, (1.1 * max_dists_p).expand(num_x, 1)], dim=1)

    weights_x = torch.full_like(dists_p[:, 0], 1.0 / num_x)[None]
    weights_y = torch.full_like(dists_p[0, :], fraction_to_match / num_y)[None]
    if fraction_to_match < 1.0:
        weights_y[:, -1] = 1.0 - fraction_to_match

    coupling = torch.from_numpy(
        ot_emd(weights_x[0].numpy(), weights_y[0].numpy(), dists_p.cpu().numpy())
    )

    if (
        coupling[:, :num_y].sum() / fraction_to_match - 1
    ).abs().item() > 1e-5:  # pragma: no cover
        raise RuntimeError("Numerical stability failed")
    wdist = ((coupling[:, :num_y] * dists_p[:, :num_y]).sum() / fraction_to_match) ** (
        1 / p
    )

    if n_perm is None and return_coupling:
        return wdist.to(dtype=dtype, device=device), coupling.to(
            dtype=dtype, device=device
        )
    elif n_perm is None:
        return wdist.to(dtype=dtype, device=device)

    xy = torch.cat([x, y], dim=0)
    scores = []
    for i in range(n_perm):
        slicing = torch.randperm(num_x + num_y)
        dists_p_0 = torch.cdist(xy[slicing[:num_x]], xy[slicing[num_x:]], p=p) ** p
        if fraction_to_match < 1.0:
            max_dists_p_0 = dists_p_0.max()
            dists_p_0 = torch.cat(
                [dists_p_0, (1.1 * max_dists_p_0).expand(num_x, 1)], dim=1
            )
        coupling_0 = torch.from_numpy(ot_emd(weights_x[0], weights_y[0], dists_p_0))

        if (
            coupling_0[:, :num_y].sum() / fraction_to_match - 1
        ).abs().item() > 1e-5:  # pragma: no cover
            raise RuntimeError("Numerical stability failed")
        scores.append(
            ((coupling_0[:, :num_y] * dists_p_0[:, :num_y]).sum() / fraction_to_match)
            ** (1 / p)
        )
    scores = torch.stack(scores)

    p_val = (wdist < scores).float().mean()
    if return_coupling:
        return (
            wdist.to(dtype=dtype, device=device),
            p_val.to(dtype=dtype, device=device),
            coupling.to(dtype=dtype, device=device),
        )
    return wdist.to(dtype=dtype, device=device), p_val.to(dtype=dtype, device=device)


class WassersteinDriftDetector(Detector):
    """Drift detector based on the Wasserstein distance.
     (see T. Viehmann: Partial Wasserstein and Maximum Mean Discrepancy distances
     for bridging the gap between outlier detection and drift detection,
     https://arxiv.org/abs/2106.01289 )

    Args:
          fraction_to_match: fraction of x probability mass to be matched
          n_perm: number of bootstrap permutations to run to compute p-value (None for not returning a p-value)
    """

    def __init__(
        self,
        *,
        return_p_value=False,
        n_perm=1000,
        fraction_to_match=1.0,
        wasserstein_p=2.0
    ):
        super().__init__(return_p_value=return_p_value)
        self.fraction_to_match = fraction_to_match
        self.wasserstein_p = wasserstein_p
        self.n_perm = n_perm
        self.n_test = None
        self.scores = None

    def fit(self, x: torch.Tensor, n_test: Optional[int] = None):
        """Record a sample as the reference distribution

        Args:
            x: The reference data
            n_test: If an int is specified, the last n_test datapoints
                will not be considered part of the reference data. Instead,
                bootstrappin using permutations will be used to determine
                the distribution under the null hypothesis at fit time.
                Future testing must then always be done with n_test elements
                to get p-values.
        """
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

            x = x.to(device="cpu")
            n_ref = x.size(0) - n_test
            with_distant_point = self.fraction_to_match < 1.0

            weights_x = torch.full((n_ref,), 1.0 / n_ref, dtype=torch.double)
            weights_y = torch.full(
                (n_test + int(with_distant_point),),
                self.fraction_to_match / n_test,
                dtype=torch.double,
            )
            if with_distant_point:
                weights_y[-1] = 1.0 - self.fraction_to_match
            p = self.wasserstein_p
            scores = []
            for i in range(self.n_perm):
                slicing = torch.randperm(x.size(0))
                dists_p_0 = (
                    torch.cdist(x[slicing[:-n_test]], x[slicing[-n_test:]], p=p) ** p
                )
                if with_distant_point:
                    max_dists_p_0 = dists_p_0.max()
                    dists_p_0 = torch.cat(
                        [dists_p_0, (1.1 * max_dists_p_0).expand(n_ref, 1)], dim=1
                    )
                coupling_0 = torch.from_numpy(ot_emd(weights_x, weights_y, dists_p_0))
                if (
                    coupling_0[:, :n_test].sum() / self.fraction_to_match - 1
                ).abs().item() > 1e-5:  # pragma: no cover
                    raise RuntimeError("Numerical stability failed")
                scores.append(
                    (
                        (coupling_0[:, :n_test] * dists_p_0[:, :n_test]).sum()
                        / self.fraction_to_match
                    )
                    ** (1 / p)
                )
            scores = torch.stack(scores)

            self.scores = scores

            self.dist_min = scores.min().double()
            mean = scores.mean() - self.dist_min
            var = scores.var().double()
            self.dist_alpha = mean ** 2 / var
            self.dist_beta = mean / var
            self.scores = scores
        return x

    def predict_shift_from_features(
        self,
        base_outputs: torch.Tensor,
        outputs: torch.Tensor,
        compute_score: bool,
        compute_p_value: bool,
        individual_samples: bool = False,
    ):
        torchdrift.utils.check(
            not individual_samples,
            "Individual samples not supported by Wasserstein distance detector",
        )
        if not compute_p_value:
            ood_score = wasserstein(
                base_outputs,
                outputs,
                p=self.wasserstein_p,
                fraction_to_match=self.fraction_to_match,
                n_perm=None,
            )
            p_value = None
        elif self.n_test is None:
            ood_score, p_value = wasserstein(
                base_outputs,
                outputs,
                p=self.wasserstein_p,
                fraction_to_match=self.fraction_to_match,
                n_perm=self.n_perm,
            )
        else:
            torchdrift.utils.check(
                self.n_test == outputs.size(0),
                "number of test samples does not match calibrated number",
            )
            ood_score = wasserstein(
                base_outputs,
                outputs,
                p=self.wasserstein_p,
                fraction_to_match=self.fraction_to_match,
                n_perm=None,
            )
            p_value = torch.igammac(
                self.dist_alpha,
                self.dist_beta * (ood_score.cpu() - self.dist_min).clamp_(min=0),
            ).to(
                outputs.device
            )  # needs PyTorch >=1.8
            # z = (ood_score - self.dist_mean) / self.dist_std
            # p_value = 0.5 * torch.erfc(z * (0.5**0.5))
            # p_value = (self.scores > ood_score).float().mean()
        return ood_score, p_value
