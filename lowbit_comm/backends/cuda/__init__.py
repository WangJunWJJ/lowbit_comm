"""Lazy access to the optional lowbit_comm CUDA extension."""

from lowbit_comm.backends.cuda.loader import (
    extension_available,
    extension_error,
    load_extension,
)

__all__ = ["extension_available", "extension_error", "load_extension"]
