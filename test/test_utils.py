import pytest
import torchdrift
import torch
import torch.utils.data


class DummyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args

    def __iter__(self):
        return iter(*self.args)


class TensorDataModule:
    def __init__(self, *args, ds_type="map"):
        self.ds_type = ds_type

        if ds_type == "map":
            self.ds = torch.utils.data.TensorDataset(*args)
        else:
            self.ds = DummyIterableDataset(*args)

    def default_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        if batch_size is None:
            batch_size = self.val_batch_size
        replacement = num_samples is not None
        if shuffle and self.ds_type == "map":
            sampler = torch.utils.data.RandomSampler(
                self.ds, replacement=replacement, num_samples=num_samples
            )
        else:
            sampler = None
        return torch.utils.data.DataLoader(
            self.ds, batch_size=batch_size, sampler=sampler
        )


@pytest.mark.parametrize("ds_type", ("map", "iterable"))
def test_fit(ds_type):
    dm_ref = TensorDataModule(torch.randn(500, 5), ds_type=ds_type)
    d = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(
        dm_ref.default_dataloader(batch_size=10),
        torch.nn.Identity(),
        [torch.nn.Identity(), d],
        num_batches=3,
        device="cpu",
    )


def test_experiment():
    torch.manual_seed(1234)
    d = torchdrift.detectors.KernelMMDDriftDetector()
    dm_ref = TensorDataModule(torch.randn(500, 5))
    dm_x = TensorDataModule(torch.randn(500, 5))
    dm_y = TensorDataModule(torch.randn(500, 5) + 1)
    experiment = torchdrift.utils.DriftDetectionExperiment(
        d,
        torch.nn.Linear(5, 5),
    )
    experiment.post_training(torch.utils.data.DataLoader(dm_ref.ds, batch_size=100))
    experiment.evaluate(dm_x, dm_y)
    experiment = torchdrift.utils.DriftDetectionExperiment(
        d, torch.nn.Linear(5, 5), ood_ratio=0.9, sample_size=10
    )
    experiment.post_training(torch.utils.data.DataLoader(dm_ref.ds, batch_size=100))
    experiment.evaluate(dm_x, dm_y)


if __name__ == "__main__":
    pytest.main([__file__])
