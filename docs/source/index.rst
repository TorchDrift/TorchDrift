.. image:: _static/logo/torchdrift-rendered.svg
   :width: 100%

TorchDrift: drift detection for PyTorch
=======================================

TorchDrift is a data and concept drift library for PyTorch. It lets you monitor your PyTorch models to see if they operate within spec.

We focus on practical application
and strive to seamlessly integrate with PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: Get started:

   installation

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   notebooks/drift_detection_on_images
   notebooks/deployment_monitoring_example

.. toctree::
   :maxdepth: 2
   :caption: Background:

   notebooks/drift_detection_overview
   notebooks/comparing_drift_detectors
   notebooks/note_on_mmd
   publications

.. toctree::
   :maxdepth: 2
   :caption: torchdrift API:

   detectors
   reducers
   data_functional
   utils


Authors
=======

TorchDrift is a joint project of Orobix Srl, Bergamo, Italy and
MathInf GmbH, Garching b. München, Germany.

The TorchDrift Team: Thomas Viehmann, Luca Antiga, Daniele Cortinovis, Lisa Lozza

Acknowledgements
================

We were inspired by

- Failing Loudly: An Empirical Study of Methods for Detecting Dataset
  Shift, NeurIPS 2019
  https://github.com/steverab/failing-loudly
- Hendrycks & Dietterich:
  Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
  ICLR 2019
  https://github.com/hendrycks/robustness/
- Van Looveren et al.: Alibi Detect https://github.com/SeldonIO/alibi-detect/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
