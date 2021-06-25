import pytest
import torchdrift
import torch


def test_detector():
    x = torch.randn(5, 5)
    d = torchdrift.detectors.Detector()
    d.fit(x)
    with pytest.raises(NotImplementedError):
        d(x)


def _test_detector_class(cls):
    torch.manual_seed(1234)
    d = cls()
    d2 = cls(return_p_value=True)
    x = torch.randn(5, 5)
    y = torch.randn(5, 5) + 1.0
    d.fit(x)
    d2.fit(x)
    assert d(x).item() < d(y).item()
    assert d.compute_p_value(x) > 0.80
    assert d.compute_p_value(y) < 0.05
    torch.manual_seed(1234)
    p1 = d.compute_p_value(y)
    torch.manual_seed(1234)
    p2 = d2(y)
    assert p1 == p2


def _test_detector_class_fit_bootstrap(cls):
    torch.manual_seed(1234)
    d = cls()
    x = torch.randn(100, 5)
    y = torch.randn(50, 5) + 1.0
    z = torch.randn(50, 5)
    d.fit(x, n_test=50)
    assert d.compute_p_value(x[:50]) > 0.80
    assert d.compute_p_value(y) < 0.05


def test_ksdetector():
    _test_detector_class(torchdrift.detectors.KSDriftDetector)


def _test_mmd_kernel(kernel):
    torch.manual_seed(1234)
    d = torchdrift.detectors.KernelMMDDriftDetector(kernel=kernel)
    x = torch.randn(5, 5)
    y = torch.randn(5, 5) + 1.0
    d.fit(x)
    assert d(x).item() < d(y).item()
    assert d.compute_p_value(x) > 0.80
    assert d.compute_p_value(y) < 0.05


def test_mmddetector():
    _test_detector_class(torchdrift.detectors.KernelMMDDriftDetector)
    _test_detector_class_fit_bootstrap(torchdrift.detectors.KernelMMDDriftDetector)
    _test_mmd_kernel(torchdrift.detectors.mmd.GaussianKernel(lengthscale=1.0))
    _test_mmd_kernel(torchdrift.detectors.mmd.ExpKernel())
    _test_mmd_kernel(torchdrift.detectors.mmd.ExpKernel(lengthscale=1.0))
    _test_mmd_kernel(torchdrift.detectors.mmd.RationalQuadraticKernel())
    _test_mmd_kernel(
        torchdrift.detectors.mmd.RationalQuadraticKernel(lengthscale=1.0, alpha=2.0)
    )

def test_wasserstein_detector():
    _test_detector_class(torchdrift.detectors.WassersteinDriftDetector)
    _test_detector_class_fit_bootstrap(torchdrift.detectors.WassersteinDriftDetector)

    def partial_wasserstein(return_p_value=False):
        return torchdrift.detectors.WassersteinDriftDetector(
            return_p_value=return_p_value,
            fraction_to_match=0.5
        )
    _test_detector_class(torchdrift.detectors.WassersteinDriftDetector)
    _test_detector_class(partial_wasserstein)
    _test_detector_class_fit_bootstrap(partial_wasserstein)

    x = torch.randn(5, 5)
    y = torch.randn(5, 5) + 1.0
    d, p, c = torchdrift.detectors.wasserstein(x, y, return_coupling=True)
    d, c = torchdrift.detectors.wasserstein(x, y, return_coupling=True, n_perm=None)


if __name__ == "__main__":
    pytest.main([__file__])
