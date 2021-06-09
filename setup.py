import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torchdrift',
    version='0.1.0.post1',
    description="Drift Detection for PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Orobix Srl and MathInf GmbH',
    url='https://torchdrift.org/',
    install_requires=['torch'],
    packages=setuptools.find_packages(),
)
