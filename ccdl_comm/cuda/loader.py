from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
import sys
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class CudaExtensionStatus:
    """Result of attempting to import the CCDL CUDA extension."""

    available: bool
    module: object | None
    reason: str | None = None
    abi_version: int | None = None
    torch_version: str | None = None
    cuda_runtime_version: str | None = None


def load_cuda_extension(
    *,
    module_name: str = "ccdl_cuda_ops",
    import_module: Callable[[str], ModuleType | object] = _import_module,
) -> CudaExtensionStatus:
    """Load the CUDA extension without leaking import failures to planners.

    ParaScale should be able to inspect CCDL availability on CPU-only or
    partially configured machines.  Missing or broken extensions are therefore
    reported as data instead of raised during import.
    """

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name in str(exc):
            return CudaExtensionStatus(
                available=False,
                module=None,
                reason=f"{module_name} is not installed",
            )
        return CudaExtensionStatus(available=False, module=None, reason=str(exc))
    except ImportError as exc:
        return CudaExtensionStatus(available=False, module=None, reason=str(exc))

    loaded_torch = sys.modules.get("torch")
    torch_version = getattr(module, "TORCH_VERSION", None)
    if torch_version is None:
        torch_version = getattr(loaded_torch, "__version__", None)
    cuda_runtime_version = getattr(module, "CUDA_RUNTIME_VERSION", None)
    if cuda_runtime_version is None:
        torch_cuda_version = getattr(loaded_torch, "version", None)
        cuda_runtime_version = getattr(torch_cuda_version, "cuda", None)
    abi_version = getattr(module, "NATIVE_WORK_ABI_VERSION", None)
    return CudaExtensionStatus(
        available=True,
        module=module,
        abi_version=int(abi_version) if abi_version is not None else None,
        torch_version=str(torch_version) if torch_version is not None else None,
        cuda_runtime_version=(
            str(cuda_runtime_version) if cuda_runtime_version is not None else None
        ),
    )
