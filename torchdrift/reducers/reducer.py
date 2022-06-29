import torch


class Reducer(torch.nn.Module):
    """Base class for reducers.

    This is a `torch.nn.Module` with an additional `fit` method.
    The usual forward is for testing after fitting."""

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        """Fits the reducer to reference data `x` and returns the reduced
        data.

        Override this in your reducer implementation."""
        raise NotImplementedError("Override fit in subclass")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the input `x` (in testing) and returns the reduced data.

        Override this in your reducer implementation."""
        raise NotImplementedError("Override forward in subclass")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for bname, b in self._buffers.items():
            if prefix + bname in state_dict and b is None:
                setattr(self, bname, state_dict[prefix + bname])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
