import torch
import tqdm


class DriftDetectionExperiment:
    def __init__(self, drift_detector, ood_ratio=1.0, sample_size=1):
        self.ood_ratio = ood_ratio
        self.sample_size = sample_size
        self.drift_detector = drift_detector

    # def extra_loss(self, ...):  add components to loss from training the detector
    def post_training(self, train_dataloader):
        "Called after training the main model"
        self.drift_detector.fit(train_dataloader)

    def evaluate(self, ind_datamodule, ood_datamodule, num_runs=50):
        device = next(self.drift_detector.model.parameters()).device
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
            ind_score = self.drift_detector.predict_shift(ind_batch[0].to(device))
            if num_ind > 0:
                drifted_batch = torch.cat([ind_batch[0][:num_ind], ood_batch[0]], dim=0)
            else:
                drifted_batch = ood_batch[0]
            drifted_score = self.drift_detector.predict_shift(drifted_batch.to(device))
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
