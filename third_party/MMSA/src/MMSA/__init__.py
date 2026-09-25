"""MMSA package with lazy top-level imports for component-only experiments."""

from .config import get_config_all, get_config_regression, get_config_tune, get_citations


def __getattr__(name):
    if name in {"MMSA_run", "MMSA_test", "SUPPORTED_DATASETS", "SUPPORTED_MODELS",
                "SENA_run", "DEMO_run"}:
        from importlib import import_module
        return getattr(import_module(f"{__name__}.run"), name)
    raise AttributeError(name)
