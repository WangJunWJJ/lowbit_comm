"""Lazy access to the optional lowbit_comm CUDA extension."""

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import (
    extension_available,
    extension_error,
    load_extension,
)
from lowbit_comm.backends.cuda.plan import CudaBackendPlan
from lowbit_comm.backends.cuda.layout import (
    FullTensorLayout,
    build_fulltensor_layout,
)

__all__ = [
    "FullTensorLayout",
    "CudaBackend",
    "CudaBackendPlan",
    "build_fulltensor_layout",
    "extension_available",
    "extension_error",
    "load_extension",
]
