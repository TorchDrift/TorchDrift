import torch

class PCAReducer(torch.nn.Module):
    def __init__(self, n_components:int = 2):
        super().__init__()
        self.n_components = n_components

    def extra_repr(self):
        return f'n_components={self.n_components}'

    def forward(self, x: torch.Tensor):
        batch, feat = x.shape
        assert min(batch, feat) >= self.n_components
        x = x - x.mean(1, keepdim=True)
        u, s, v = x.svd()
        comp = v[:, :self.n_components]
        reduced = x @ comp
        return reduced
