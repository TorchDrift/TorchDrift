import setuptools

setuptools.setup(
  name='torchdrift',
  version='0.1.0',
  description="Drift Detection for PyTorch",
  author='Orobix Srl and MathInf GmbH',
  install_requires=['torch'],
  packages=setuptools.find_packages()
  )
