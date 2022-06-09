import torch
from . import Reducer
import torchdrift.utils


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
        self.register_buffer("mean", None)
        self.register_buffer("comp", None)

    def extra_repr(self) -> str:
        return f"n_components={self.n_components}"

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        batch, feat = x.shape
        torchdrift.utils.check(
            min(batch, feat) >= self.n_components,
            "need number of samples and size of feature to be at least the number of components",
        )
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
