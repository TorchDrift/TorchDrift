from typing import Optional, Callable, Union, Iterable, List
import typing
from ..reducers import Reducer
from ..detectors import Detector

import tqdm
import torch

def fit(
        dl: torch.utils.data.DataLoader,
        feature_extractor: torch.nn.Module,
        reducers_detectors: Union[Reducer, Detector, List[Union[Reducer, Detector]]],
        num_batches: Optional[int] = None,
        device: Union[torch.device, str, None] = None
    ):
    """Train drift detector on reference distribution.
"""
    if not isinstance(reducers_detectors, typing.Iterable):
        reducers_detectors = [reducers_detectors]
    feature_extractor.eval()  # careful about test time dropout
    if device is None:
        device = next(feature_extractor.parameters()).device

    all_outputs = []
    # dl = torch.utils.data.DataLoader(ref_ds, batch_size=batch_size, shuffle=True)
    nb = len(dl)
    if num_batches is not None:
        nb = min(nb, num_batches)
    for i, (b, _) in tqdm.tqdm(zip(range(nb), dl), total=nb):
        with torch.no_grad():
            all_outputs.append(feature_extractor(b.to(device)))
    all_outputs = torch.cat(all_outputs, dim=0)

    for m in reducers_detectors:
        if hasattr(m, 'fit'):
            all_outputs = m.fit(all_outputs)
        else:
            all_outputs = m(all_outputs)
