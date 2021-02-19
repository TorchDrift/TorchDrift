import pytest
import torchdrift
import torch
import sklearn.decomposition

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
    assert (d(x).item() < d(y).item())
    assert d.compute_p_value(x) > 0.80
    assert d.compute_p_value(y) < 0.05
    assert d.compute_p_value(y) == d2(y)

def test_ksdetector():
    _test_detector_class(torchdrift.detectors.KSDriftDetector)

def test_mmddetector():
    _test_detector_class(torchdrift.detectors.KernelMMDDriftDetector)

if __name__ == "__main__":
    pytest.main([__file__])
