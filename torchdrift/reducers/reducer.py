import torch

class Reducer(torch.nn.Module):
    """Base class for reducers"""

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Override fit in subclass")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Override forward in subclass")
