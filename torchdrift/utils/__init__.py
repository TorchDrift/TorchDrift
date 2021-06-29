from .experiments import DriftDetectionExperiment
from .fit import fit


def check(check, message, error_class=RuntimeError):
    """tests `check` and raises `RuntimeError` with `message` if false"""
    if not check:
        raise error_class(message)
