import setuptools

setuptools.setup(
  name='torchdrift',
  version='0.0.1',
  description="Drift Detection for PyTorch",
  author='Orobix Srl and MathInf GmbH',
  install_requires=['torch', 'torchvision', 'pytorch_lightning'],
  packages=setuptools.find_packages()
  )
