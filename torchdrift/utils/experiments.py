import torch
import tqdm
from .fit import fit


class DriftDetectionExperiment:
    """An experimental setup to explore the ROC of drift detection setups

    This tests a setup based on a drift detector and a feature extractor (the latter including reducers).

    The detector is fitted with `post_training`.

    Then given datamodules for in-distribution and out-of-distribution, non-drifted and drifted batches are constructed. The test batches have the `sample_size` given as a constructor parameter and a fraction `ood_ratio` samples (rounded up) are from the out-of-distribution datamodule.

    The datamodules are expected to provide a `default_dataloader` method taking
    `batch_size` and `num_samples` arguments (see the examples for details)."""

    def __init__(self, drift_detector, feature_extractor, ood_ratio=1.0, sample_size=1):
        self.ood_ratio = ood_ratio
        self.sample_size = sample_size
        self.drift_detector = drift_detector
        self.feature_extractor = feature_extractor

    # def extra_loss(self, ...):  add components to loss from training the detector
    def post_training(self, train_dataloader):
        "Called after training the main model, fits the drift detector."
        fit(train_dataloader, self.feature_extractor, self.drift_detector)

    def evaluate(self, ind_datamodule, ood_datamodule, num_runs=50):
        """runs the experiment (`num_runs` inputs)

        Returns: auc, (fp, tp)
            auc: Area-under-Curve score

            fp, tp: False positive and true positive rates to plot the ROC curve.

        """
        device = next(self.feature_extractor.parameters()).device
        # numbers for drifted scenarios
        num_ood = int(self.sample_size * self.ood_ratio)
        num_ind = self.sample_size - num_ood
        # TODO: what to do if we cannot fit it all in one batch?
        ind_dl = ind_datamodule.default_dataloader(
            batch_size=self.sample_size, num_samples=self.sample_size * num_runs
        )
        ood_dl = ood_datamodule.default_dataloader(
            batch_size=num_ood, num_samples=num_ood * num_runs
        )
        assert num_ood > 0 and num_ind >= 0
        all_drifted_scores = []
        all_ind_scores = []
        for r, (ind_batch, ood_batch) in tqdm.tqdm(
            enumerate(zip(ind_dl, ood_dl)), total=num_runs
        ):
            with torch.no_grad():
                ind_feat = self.feature_extractor(ind_batch[0].to(device))
                ind_score = self.drift_detector(ind_feat)
            if num_ind > 0:
                drifted_batch = torch.cat([ind_batch[0][:num_ind], ood_batch[0]], dim=0)
            else:
                drifted_batch = ood_batch[0]
            with torch.no_grad():
                drifted_feat = self.feature_extractor(drifted_batch.to(device))
                drifted_score = self.drift_detector(drifted_feat)
            all_ind_scores.append(ind_score)
            all_drifted_scores.append(drifted_score)

        labels = torch.cat(
            [
                torch.ones(len(all_drifted_scores), dtype=torch.bool),
                torch.zeros(len(all_ind_scores), dtype=torch.bool),
            ]
        )
        scores = torch.tensor(all_drifted_scores + all_ind_scores)
        self.all_drifted_scores = all_drifted_scores
        self.all_ind_scores = all_ind_scores
        # compute ROC curve
        threshold = torch.linspace(scores.max(), scores.min(), len(scores))
        predictions = scores[None] >= threshold[:, None]
        tp = (predictions & labels[None]).sum(1).float() / len(all_drifted_scores)
        fp = (predictions & ~labels[None]).sum(1).float() / len(all_ind_scores)
        fp_diff = fp[1:] - fp[:-1]
        tp_avg = (tp[1:] + tp[:-1]) / 2
        auc = (fp_diff * tp_avg).sum()
        return auc, (fp, tp)
