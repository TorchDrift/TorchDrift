detectors
=========

The detectors are the central components of TorchDrift.

The inputs are expected to be 2d-`Tensor` s: `batch` x `feature_dim`.

.. automodule:: torchdrift.detectors
   :imported-members:
   :members:
   :undoc-members:

Kernels for MMD
---------------
   .. autoclass:: torchdrift.detectors.mmd.Kernel

   .. autoclass:: torchdrift.detectors.mmd.GaussianKernel

   .. autoclass:: torchdrift.detectors.mmd.ExpKernel

   .. autoclass:: torchdrift.detectors.mmd.RationalQuadraticKernel
