"""Load model families only when requested."""


def __getattr__(name):
    if name == "AMIO":
        from .AMIO import AMIO
        return AMIO
    raise AttributeError(name)
