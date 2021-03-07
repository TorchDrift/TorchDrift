from .experiments import DriftDetectionExperiment
from .fit import fit


def check(check, message):
    """tests `check` and raises `RuntimeError` with `message` if false"""
    if not check:
        raise RuntimeError(message)
