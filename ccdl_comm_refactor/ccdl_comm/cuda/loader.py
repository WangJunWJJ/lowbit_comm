from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class CudaExtensionStatus:
    """Result of attempting to import the CCDL CUDA extension."""

    available: bool
    module: object | None
    reason: str | None = None


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

    return CudaExtensionStatus(available=True, module=module)
