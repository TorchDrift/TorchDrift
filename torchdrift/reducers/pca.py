import torch
from . import Reducer


class PCAReducer(Reducer):
    """Reduce dimensions using PCA.

    This nn.Modue subclass reduces the dimensions of the inputs
    specified by `n_components`.

    The input is a 2-dimensional `Tensor` of size `batch` x `features`,
    the output is a `Tensor` of size `batch` x `n_components`.
    """

    def __init__(self, n_components: int = 2):
        super().__init__()
        self.n_components = n_components

    def extra_repr(self) -> str:
        return f"n_components={self.n_components}"

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        batch, feat = x.shape
        assert min(batch, feat) >= self.n_components
        self.mean = x.mean(0, keepdim=True)
        x = x - self.mean
        u, s, v = x.svd()
        self.comp = v[:, : self.n_components]
        reduced = x @ self.comp
        return reduced

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        reduced = x @ self.comp
        return reduced
