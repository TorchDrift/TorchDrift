from typing import Optional, Union, List
import typing
from ..reducers import Reducer
from ..detectors import Detector

import tqdm
import torch


def fit(
    dl: torch.utils.data.DataLoader,
    feature_extractor: torch.nn.Module,
    reducers_detectors: Union[Reducer, Detector, List[Union[Reducer, Detector]]],
    *,
    num_batches: Optional[int] = None,
    device: Union[torch.device, str, None] = None
):
    """Train drift detector on reference distribution.

    The dataloader `dl` should provide the reference distribution. Optionally you can limit the number of batches sampled from the dataloader with `num_batches`.

    The `feature extractor` can be any module be anything that does not need to be fit.

    The reducers and detectors should be passed (in the order they should be applied, one takes the output from the previous) as a list. A single detector or reducer can also be passed.

    If you provide a `device`, data is moved there before running through the
    feature extractor, otherwise the functions try to infer the device from the `feature_extractor`."""
    if not isinstance(reducers_detectors, typing.Iterable):
        reducers_detectors = [reducers_detectors]
    feature_extractor.eval()  # careful about test time dropout
    if device is None:
        device = next(feature_extractor.parameters()).device

    all_outputs = []
    # dl = torch.utils.data.DataLoader(ref_ds, batch_size=batch_size, shuffle=True)

    if hasattr(dl.dataset, "__len__"):
        nb = len(dl)
        if num_batches is not None:
            nb = min(nb, num_batches)
        total = nb
    else:
        total = None

    for i, b in enumerate(tqdm.tqdm(dl, total=total)):
        if num_batches is not None and i >= num_batches:
            break

        if not isinstance(b, torch.Tensor):
            b = b[0]
        with torch.no_grad():
            all_outputs.append(feature_extractor(b.to(device)))

    all_outputs = torch.cat(all_outputs, dim=0)

    for m in reducers_detectors:
        if hasattr(m, "fit"):
            all_outputs = m.fit(all_outputs)
        else:
            all_outputs = m(all_outputs)
