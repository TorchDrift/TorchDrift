from typing import Optional

import torch
import tqdm
import flash

from . import kernel_mmd


class KernelMMDDriftDetector:
    def __init__(
        self,
        model: torch.nn.Module,
        data_pipeline: Optional[flash.core.data.DataPipeline] = None,
    ):
        self.model = model
        self.data_pipeline = data_pipeline
        self.base_outputs = None

    def fit(
        self,
        ref_ds: torch.utils.data.Dataset,
        data_pipeline: Optional[flash.core.data.DataPipeline] = None,
        batch_size: int = 32,
        num_batches: Optional[int] = None,
    ):
        self.model.eval()  # careful about test time dropout
        if data_pipeline is None:
            data_pipeline = self.data_pipeline
            assert (
                data_pipeline is not None
            ), "please pass data_pipeline to constructor or .fit"
        all_outputs = []
        dl = torch.utils.data.DataLoader(
            ref_ds,
            batch_size=batch_size,
            collate_fn=data_pipeline.collate_fn,
            shuffle=True,
        )
        nb = len(dl)
        if num_batches is not None:
            nb = min(nb, num_batches)
        for i, (b, _) in tqdm.tqdm(zip(range(nb), dl), total=nb):
            # predict puts model in eval, does no_grad
            o = data_pipeline.uncollate_fn(
                self.model.predict(b, skip_collate_fn=True, data_pipeline=data_pipeline)
            )
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, dim=0)
        self.base_outputs = all_outputs

    def predict_shift(
        self,
        input_batch: torch.Tensor,
        data_pipeline: Optional[flash.core.data.DataPipeline] = None,
        individual_samples: bool = False,
        skip_collate_fn=True,
    ):
        self.model.eval()  # careful about test time dropout
        if data_pipeline is None:
            data_pipeline = self.data_pipeline
        assert self.base_outputs is not None, "Please call fit before predict_shift"
        assert not individual_samples, "Individual samples not supported"
        outputs = self.model.predict(
            input_batch, data_pipeline=data_pipeline, skip_collate_fn=skip_collate_fn
        )
        self.last_outputs = outputs
        ood_score = kernel_mmd(
            outputs, self.base_outputs, n_perm=None
        )  # we have higher == more abnormal
        return ood_score
