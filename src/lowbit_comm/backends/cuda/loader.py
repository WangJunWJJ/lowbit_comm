"""Safe, lazy loading of the optional CUDA extension."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from types import ModuleType
from typing import Callable


@dataclass(frozen=True, slots=True)
class CudaExtensionStatus:
    available: bool
    module: object | None
    reason: str | None = None
    abi_version: int | None = None


def load_cuda_extension(
    *,
    module_name: str = "lowbit_comm_cuda_ops",
    import_module: Callable[[str], ModuleType | object] = _import_module,
) -> CudaExtensionStatus:
    """Return extension availability without breaking CPU-only imports."""

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name in str(error):
            reason = f"{module_name} is not installed"
        else:
            reason = str(error)
        return CudaExtensionStatus(False, None, reason)
    except (ImportError, OSError) as error:
        return CudaExtensionStatus(False, None, str(error))

    abi_version = getattr(module, "NATIVE_WORK_ABI_VERSION", None)
    return CudaExtensionStatus(
        True,
        module,
        abi_version=int(abi_version) if abi_version is not None else None,
    )
