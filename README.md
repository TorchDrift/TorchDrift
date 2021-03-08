![](docs/source/_static/logo/torchdrift-rendered.svg)

# TorchDrift: drift detection for PyTorch

TorchDrift is a data and concept drift library for PyTorch. It lets you monitor your PyTorch models to see if they operate within spec.

We focus on practical application
and strive to seamlessly integrate with PyTorch.

# Installation

To install the latest release version use

```
pip install torchdrift
```

To get the latest and greatest install from git with
```
pip install git+https://github.com/torchdrift/torchdrift/
```

# Documentation

Our documentation is at [TorchDrift.org](https://torchdrift.org/).

# Examples

Check out [our worked example](https://torchdrift.org/notebooks/drift_detection_on_images.html) with an ImageNet-type classifier.

If you have a model (without head) as the feature extractor and a training dataloader you can fit the reference distribution as

```python
drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
torchdrift.utils.fit(train_dataloader, feature_extractor, drift_detector)
```

and then check drifts with

```python
features = feature_extractor(inputs)
score = drift_detector(features)
p_val = drift_detector.compute_p_value(features)

if p_val < 0.01:
    raise RuntimeError("Drifted Inputs")
```

Also check out our [deployment example](https://torchdrift.org/notebooks/deployment_monitoring_example.html) for integration of TorchDrift into
inference with a model.

# Authors

TorchDrift is a joint project of Orobix Srl, Bergamo, Italy and
MathInf GmbH, Garching b. München, Germany.

The TorchDrift Team: Thomas Viehmann, Luca Antiga, Daniele Cortinovis, Lisa Lozza

# Acknowledgements

We were inspired by

- Failing Loudly: An Empirical Study of Methods for Detecting Dataset
  Shift, NeurIPS 2019
  https://github.com/steverab/failing-loudly
- Hendrycks & Dietterich:
  Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
  ICLR 2019
  https://github.com/hendrycks/robustness/
- Van Looveren et al.: Alibi Detect https://github.com/SeldonIO/alibi-detect/
