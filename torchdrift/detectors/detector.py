import torch
import torchdrift.utils


class Detector(torch.nn.Module):
    """Detector class.

    The detector is is a `nn.Module` subclass that, after fitting, performs a drift test when called and returns a score or p-value.

        Constructor Args:
                return_p_value (bool): If set, forward returns a p-value (estimate) instead of the raw test score.
    """

    def __init__(self, *, return_p_value: bool = False):
        super().__init__()
        self.register_buffer("base_outputs", None)
        self.return_p_value = return_p_value

    def fit(self, x: torch.Tensor):
        """Record a sample as the reference distribution"""
        self.base_outputs = x.detach()
        return x

    def predict_shift_from_features(
        self,
        base_outputs: torch.Tensor,
        outputs: torch.Tensor,
        compute_score: bool,
        compute_p_value: bool,
        individual_samples: bool = False,
    ) -> torch.Tensor:
        """stub to be overridden by subclasses"""
        raise NotImplementedError("Override predict_shift_from_features in detectors")

    def compute_p_value(self, inputs: torch.Tensor) -> torch.Tensor:
        """Performs a statistical test for drift and returns the p-value.

        This method calls `predict_shift_from_features` under the hood, so you only need to override that when subclassing."""
        torchdrift.utils.check(
            self.base_outputs is not None, "Please call fit before compute_p_value"
        )
        _, p_value = self.predict_shift_from_features(
            self.base_outputs, inputs, compute_score=False, compute_p_value=True
        )
        return p_value

    def forward(
        self, inputs: torch.Tensor, individual_samples: bool = False
    ) -> torch.Tensor:
        """Performs a statistical test for drift and returns the score or, if `return_p_value` has been set in the constructor, the p-value.

        This method calls `predict_shift_from_features` under the hood, so you only need to override that when subclassing."""
        torchdrift.utils.check(
            self.base_outputs is not None, "Please call fit before predict_shift"
        )
        ood_score, p_value = self.predict_shift_from_features(
            self.base_outputs,
            inputs,
            compute_score=not self.return_p_value,
            compute_p_value=self.return_p_value,
            individual_samples=individual_samples,
        )
        if self.return_p_value:
            return p_value
        return ood_score

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
