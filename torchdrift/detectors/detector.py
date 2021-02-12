from typing import Optional, Callable
import tqdm

import torch

class DriftDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base_outputs', None)

    def fit(
        self,
        ref_ds: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int = 32,
        num_batches: Optional[int] = None,
    ):
        model.eval()  # careful about test time dropout
        device = next(model.parameters()).device
        all_outputs = []
        dl = torch.utils.data.DataLoader(ref_ds, batch_size=batch_size, shuffle=True)
        nb = len(dl)
        if num_batches is not None:
            nb = min(nb, num_batches)
        for i, (b, _) in tqdm.tqdm(zip(range(nb), dl), total=nb):
            with torch.no_grad():
                all_outputs.append(model(b.to(device)))
        all_outputs = torch.cat(all_outputs, dim=0)
        self.base_outputs = all_outputs

    def predict_shift_from_features(self, base_outputs: torch.Tensor, outputs: torch.Tensor, individual_samples: bool = False):
        raise NotImplementedError("Override predict_shift_from_features in detectors")

    def forward(
            self, inputs: torch.Tensor,
            individual_samples: bool = False
    ):
        assert self.base_outputs is not None, "Please call fit before predict_shift"
        ood_score = self.predict_shift_from_features(self.base_outputs, inputs, individual_samples=individual_samples)
        return ood_score
