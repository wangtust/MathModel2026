"""Lazily expose upstream models so MulT does not import unused dependencies."""

__all__ = ["EF_LSTM", "LF_DNN", "LMF", "TFN", "MFN", "Graph_MFN",
           "BERT_MAG", "MULT", "MISA", "MFM", "MMIM", "MCTN", "CENET", "ALMT"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        return getattr(import_module(f"{__name__}.{name}"), name)
    raise AttributeError(name)
