from typing import Optional

import torch
import tqdm


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


class KernelMMDDriftDetector:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.base_outputs = None
        self.model.eval()  # careful about test time dropout

    def fit(
        self,
        ref_ds: torch.utils.data.Dataset,
        batch_size: int = 32,
        num_batches: Optional[int] = None,
    ):
        device = next(self.model.parameters()).device
        all_outputs = []
        dl = torch.utils.data.DataLoader(ref_ds, batch_size=batch_size, shuffle=True)
        nb = len(dl)
        if num_batches is not None:
            nb = min(nb, num_batches)
        for i, (b, _) in tqdm.tqdm(zip(range(nb), dl), total=nb):
            # predict puts model in eval, does no_grad
            o = self.model.predict(b.to(device))
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, dim=0)
        self.base_outputs = all_outputs

    def predict_shift(
        self, input_batch: torch.Tensor, individual_samples: bool = False
    ):
        assert self.base_outputs is not None, "Please call fit before predict_shift"
        assert (
            not individual_samples
        ), "Individual samples not supported by MMD detector"
        outputs = self.model.predict(input_batch)
        self.last_outputs = outputs
        ood_score = kernel_mmd(
            outputs, self.base_outputs, n_perm=None
        )  # we have higher == more abnormal
        return ood_score
