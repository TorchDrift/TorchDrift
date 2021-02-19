import pytest
import torchdrift
import torch
import scipy.ndimage

# these tests could be made more specific, eventually

def test_gaussian_noise():
    a = torch.rand(4, 3, 32, 32)
    b = torch.rand(1, 3, 200, 200)
    a1 = torchdrift.data.functional.gaussian_noise(a)
    b1 = torchdrift.data.functional.gaussian_noise(b)
    assert (a1 == a1.clamp(min=0, max=1)).all()
    assert (a-a1).std() < (b-b1).std()

def test_shot_noise():
    a = torch.rand(4, 3, 100, 100)
    a1 = torchdrift.data.functional.shot_noise(a, severity=4)
    a2 = torchdrift.data.functional.gaussian_noise(a, severity=4)
    assert (a1-a).abs().max() < (a2-a).abs().max()
    
def test_impulse_noise():
    a = torch.rand(4, 3, 100, 100)/2+0.25
    a1 = torchdrift.data.functional.impulse_noise(a, severity=4)
    assert a1.min().abs() + (1 - a1.max()).abs() < 1e-6

def test_speckle_noise():
    a = torch.rand(4, 3, 100, 100)/2+0.25
    a1 = torchdrift.data.functional.speckle_noise(a, severity=4)
    assert a1.min().abs() + (1 - a1.max()).abs() < 1e-6

def test_gaussian_blur():
    a = torch.rand(1, 3, 300, 300)
    a1 = torchdrift.data.functional.gaussian_blur(a, severity=5)
    a2 = scipy.ndimage.gaussian_filter(a, [0, 0, 6, 6])
    assert ((a1-a2)[:,:, 32:-32, 32:-32]).max().abs() < 1e-2


if __name__ == "__main__":
    pytest.main([__file__])
