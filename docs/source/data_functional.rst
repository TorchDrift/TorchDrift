data.functional
===============

The corruption functions are modelled after
D. Hendrycks and T. Dietterich: *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ICLR 2019)
and the code provided by their authors in https://github.com/hendrycks/robustness/ but our implementation is expected to differ in details from theirs.

It is important to use 0..1-values image tensors as inputs.
This means that you need to do the normalization after applying these.
In the traditional default PyTorch/TorchVision pipeline, images are normalized in the dataset, which may be too early. In our examples, we like to move the normalization to the beginning of the model itself or in a separate augmentation model.


.. automodule:: torchdrift.data.functional
   :imported-members:
   :members:
   :undoc-members:
