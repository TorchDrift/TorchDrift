from typing import Optional
import torch

from .mmd import GaussianKernel
from .detector import Detector
import torchdrift
import warnings

def partial_kernel_mmd_twostage(x, y, n_perm=None,
                                kernel=GaussianKernel(),
                                fraction_to_match=1.0, wasserstein_p=2.0):
    """Partial kernel MMD using a Wasserstein coupling to obtain the weight for the reference.
    """
    torchdrift.utils.check(
        n_perm is None,
        "Bootstrapping within partial MMD is not implemented, use bootstrap during fit",
        error_class=NotImplementedError
    )
    n, d = x.shape
    m, d2 = y.shape
    if fraction_to_match < 1.0:
        _, coupling = torchdrift.detectors.wasserstein(
            x, y,
            fraction_to_match=fraction_to_match, return_coupling=True, n_perm=None,
            p=wasserstein_p)
        w = coupling[:, :-1].sum(1).to(device=x.device, dtype=x.dtype) / fraction_to_match
    else:
        w = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)

    torchdrift.utils.check(d == d2, "feature dimension mismatch")
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    k = kernel(dists)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    mmd = (w @ k_x) @ w + k_y.sum() / (m * m) - 2 * (w @ k_xy).sum() / m
    return mmd

def partial_kernel_mmd_qp(x, y, n_perm=None, kernel=GaussianKernel(),
                          fraction_to_match=1.0):
    """Partial Kernel MMD using quadratic programming.

    This is very slow and mainly intended for reference purposes.
    You need to install qpsolvers to use this function."""

    torchdrift.utils.check(
        n_perm is None,
        "Bootstrapping within partial MMD is not implemented, use bootstrap during fit",
        error_class=NotImplementedError
    )
    import qpsolvers
    n, d = x.shape
    m, d2 = y.shape
    torchdrift.utils.check(d == d2, "feature dimension mismatch")
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    k = kernel(dists.double())
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]

    v = torch.full((m,), 1 / m, dtype=k_y.dtype, device=k_y.device)
    R = torch.cholesky(k_x, upper=True)
    d = torch.inverse(R.t()) @ (k_xy.sum(1) / m)
    lb = torch.zeros((n,), dtype=k_x.dtype, device=k_x.device)
    ub = torch.full((n,), 1.0 / (n * fraction_to_match), dtype=k_x.dtype, device=k_x.device)
    w = qpsolvers.solve_ls(R.cpu().numpy(), d.cpu().numpy(),
                           lb=lb.cpu().numpy(), ub=ub.cpu().numpy(),
                           A=torch.ones((1, n,), dtype=R.dtype).numpy(),
                           b=torch.ones((1,), dtype=R.dtype).numpy())
    torchdrift.utils.check(
        w is not None,
        'QP failed to find a solution (numerical accuracy with the bounds?)'
    )
    w = torch.as_tensor(w, device=k_x.device, dtype=k_x.dtype)
    mmd = (w @ k_x) @ w + k_y.sum() / (m * m) - 2 * (w @ k_xy).sum() / m
    return mmd

def partial_kernel_mmd_approx(
        x, y,
        fraction_to_match=1.0,
        kernel=GaussianKernel(),
        n_perm=None):
    torchdrift.utils.check(
        n_perm is None,
        "Bootstrapping within partial MMD is not implemented, use bootstrap during fit",
        error_class=NotImplementedError
    )
    rng = torch.Generator(device=x.device).manual_seed(1234)
    n, d = x.shape
    m, d2 = y.shape
    torchdrift.utils.check(d == d2, "feature dimension mismatch")
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    k = kernel(dists.double())
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]

    w = torch.full((n,), 1.0 / n, dtype=k_x.dtype, device=k_x.device, requires_grad=False)
    mmd = (w @ k_x) @ w + k_y.sum() / (m * m) - 2 * (w @ k_xy).sum() / m

    for i in range(100):
        r = torch.rand((), device=x.device, dtype=x.dtype, generator=rng) + 0.5
        grad_mmd = (k_x @ w) - (k_xy).mean(1)
        grad_mmd_min = grad_mmd[(w < 1.0 / (n * fraction_to_match))]
        if grad_mmd_min.size(0) > 0:
            grad_mmd_min = grad_mmd_min.min()
        else:
            grad_mmd_min = torch.zeros_like(r)
        grad_mmd_max = grad_mmd[(w > 0)]
        if grad_mmd_max.size(0) > 0:
            grad_mmd_max = grad_mmd_max.max()
        else:  # pragma: no cover
            grad_mmd_max = torch.zeros_like(r)
        active_mask = (((w > 0) | (grad_mmd < grad_mmd_min * r))
                       & ((w < 1.0 / (n * fraction_to_match)) | (grad_mmd > grad_mmd_max * r)))

        H_mmd_active = k_x[active_mask][:, active_mask]
        if H_mmd_active.size(0) == 0:
            continue
        u = torch.cholesky(H_mmd_active)
        Hinvgr = torch.cholesky_solve(grad_mmd[active_mask][:, None], u).squeeze(1)

        w_active = w[active_mask]

        Hinvgr -= Hinvgr.mean()
        Hinvgr_full = torch.zeros_like(w)
        Hinvgr_full[active_mask] = Hinvgr

        step = 1.0
        for j in range(5):
            w_cand = w.clone()
            w_cand -= step * Hinvgr_full

            w_cand.clamp_(min=0, max=1.0 / (n * fraction_to_match))
            w_cand /= w_cand.sum()
            mmd_cand = (w_cand @ k_x) @ w_cand + k_y.sum() / (m * m) - 2 * (w_cand @ k_xy).sum() / m
            is_lower = (mmd_cand < mmd)
            mmd = torch.where(is_lower, mmd_cand, mmd)
            w = torch.where(is_lower, w_cand, w)
            step /= 5

        grad_mmd = 2 * (k_x @ w) - 2 * (k_xy).mean(1)
        grad_mmd_min = grad_mmd[(w < 1.0 / (n * fraction_to_match))]
        if grad_mmd_min.size(0) > 0:
            grad_mmd_min = grad_mmd_min.min()
        else:
            grad_mmd_min = torch.zeros_like(r)
        grad_mmd_max = grad_mmd[(w > 0)]
        if grad_mmd_max.size(0) > 0:
            grad_mmd_max = grad_mmd_max.max()
        else:  # pragma: no cover
            grad_mmd_max = torch.zeros_like(r)
        active_mask = (((w > 0) | (grad_mmd < grad_mmd_min * r)) &
                       ((w < 1.0 / (n * fraction_to_match)) | (grad_mmd > grad_mmd_max * r)))
        step = 1e-1
        for j in range(5):
            w_candnd = w.clone()
            grad_mmd_x = grad_mmd.clone()
            grad_mmd_x = torch.where(active_mask,
                                     grad_mmd_x,
                                     torch.zeros((), device=grad_mmd_x.device, 
                                                 dtype=grad_mmd_x.dtype))
            grad_mmd_x = torch.where(active_mask,
                                     grad_mmd_x,
                                     grad_mmd_x - grad_mmd_x.mean())
            w_cand -= step * grad_mmd_x
            w_cand.clamp_(min=0, max=1.0 / (n * fraction_to_match))
            w_cand /= w_cand.sum()
            mmd_cand = (w_cand @ k_x) @ w_cand + k_y.sum() / (m * m) - 2 * (w_cand @ k_xy).sum() / m
            is_lower = (mmd_cand < mmd)
            mmd = torch.where(is_lower, mmd_cand, mmd)
            w = torch.where(is_lower, w_cand, w)
            step = step / 5

    return mmd

class PartialKernelMMDDriftDetector(Detector):
    """Drift detector based on the partial MMD Distance.
    (see T. Viehmann: Partial Wasserstein and Maximum Mean Discrepancy distances
    for bridging the gap between outlier detection and drift detection,
    https://arxiv.org/abs/2106.01289 )
    Note: We recommend using dtype double as input for now.

    Args:
         fraction_to_match: fraction of x probability mass to be matched
         n_perm: number of bootstrap permutations to run to compute p-value (None for not returning a p-value)
         method: PartialKernelMMDDriftDetector.METHOD_TWOSTAGE, METHOD_APPROX, or METHOD_QP
    """

    METHOD_TWOSTAGE = 1
    METHOD_APPROX = 2
    METHOD_QP = 3

    def __init__(
            self, *, return_p_value=False, n_perm=1000, fraction_to_match=1.0,
            kernel=GaussianKernel(),
            method=METHOD_TWOSTAGE,
    ):
        super().__init__(return_p_value=return_p_value)
        self.fraction_to_match = fraction_to_match
        self.kernel = kernel
        self.n_perm = n_perm
        self.n_test = None
        self.scores = None
        if method == PartialKernelMMDDriftDetector.METHOD_TWOSTAGE:
            self.partial_mmd = partial_kernel_mmd_twostage
        elif method == PartialKernelMMDDriftDetector.METHOD_APPROX:
            self.partial_mmd = partial_kernel_mmd_approx
        elif method == PartialKernelMMDDriftDetector.METHOD_QP:
            self.partial_mmd = partial_kernel_mmd_qp
        else:  # pragma: no cover
            raise RuntimeError("Invalid Partial MMD method")


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
            torchdrift.utils.check(0 < n_test < x.size(0), "n_test must be strictly between 0 and the number of samples")
            self.n_test = n_test
            self.base_outputs = x[:-n_test]

            n_ref = x.size(0) - n_test
            with_distant_point = self.fraction_to_match < 1.0

            scores = []
            for i in range(self.n_perm):
                slicing = torch.randperm(x.size(0))
                scores.append(self.partial_mmd(x[slicing[:-n_test]], x[slicing[-n_test:]], fraction_to_match=self.fraction_to_match, kernel=self.kernel))
            scores = torch.stack(scores)

            self.scores = scores

            self.dist_min = scores.min().double()
            mean = scores.mean() - self.dist_min
            var = scores.var().double()
            self.dist_alpha = mean**2 / var
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
            not individual_samples, "Individual samples not supported by Wasserstein distance detector"
        )
        if not compute_p_value:
            ood_score = self.partial_mmd(
                base_outputs, outputs, fraction_to_match=self.fraction_to_match,
                n_perm=None,
            )
            p_value = None
        else:
            torchdrift.utils.check(
                self.n_test is not None,
                "Bootstrapping within partial MMD is not implemented, use bootstrap during fit",
                error_class=NotImplementedError
            )
            torchdrift.utils.check(self.n_test == outputs.size(0),
                                   "number of test samples does not match calibrated number")
            ood_score = self.partial_mmd(
                base_outputs, outputs, fraction_to_match=self.fraction_to_match,
                n_perm=None)
            p_value = torch.igammac(self.dist_alpha, self.dist_beta * (ood_score - self.dist_min).clamp_(min=0))  # needs PyTorch >=1.8
            # z = (ood_score - self.dist_mean) / self.dist_std
            # p_value = 0.5 * torch.erfc(z * (0.5**0.5))
            # p_value = (self.scores > ood_score).float().mean()
        return ood_score, p_value
