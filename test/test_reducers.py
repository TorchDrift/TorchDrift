import pytest
import torchdrift
import torch
import sklearn.decomposition


def test_pca():
    pca = torchdrift.reducers.PCAReducer(n_components=2)
    assert "n_components" in str(pca)
    a = torch.randn(100, 50, dtype=torch.double)
    red = pca.fit(a)
    pca_ref = sklearn.decomposition.PCA(n_components=2)
    red_ref = torch.from_numpy(pca_ref.fit_transform(a))
    # need to find a way to deal with signs
    torch.testing.assert_allclose(red.abs(), red_ref.abs())
    b = torch.randn(25, 50, dtype=torch.double)
    red2 = pca(b)
    red2_ref = torch.from_numpy(pca_ref.transform(b))

def test_reducer_load_save():
    pca = torchdrift.reducers.PCAReducer(n_components=2)
    a = torch.randn(100, 50, dtype=torch.double)
    red = pca.fit(a)
    pca2 = torchdrift.reducers.PCAReducer(n_components=2)
    pca2.load_state_dict(pca.state_dict())
    red2 = pca2(a)

def test_reducer():
    x = torch.randn(5, 5)
    r = torchdrift.reducers.Reducer()
    with pytest.raises(NotImplementedError):
        r.fit(x)
    with pytest.raises(NotImplementedError):
        r(x)


if __name__ == "__main__":
    pytest.main([__file__])
