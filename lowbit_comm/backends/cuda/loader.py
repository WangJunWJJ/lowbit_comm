"""Load the optional CUDA extension only when a CUDA backend needs it."""

from __future__ import annotations

import importlib
from types import ModuleType

from lowbit_comm.core.errors import CompileError


CUDA_ABI_VERSION = 1
_module: ModuleType | None = None
_failure: CompileError | None = None


def load_extension() -> ModuleType:
    """Return the ABI-compatible CUDA extension or raise a stable error."""
    global _module, _failure
    if _module is not None:
        return _module
    if _failure is not None:
        raise _failure
    try:
        candidate = importlib.import_module("lowbit_comm._C")
        if (
            type(candidate.abi_version()) is not int
            or candidate.abi_version() != CUDA_ABI_VERSION
        ):
            raise CompileError("CUDA extension ABI is incompatible.")
        _module = candidate
        return candidate
    except CompileError as error:
        _failure = error
        raise
    except Exception as error:
        _failure = CompileError("CUDA extension is unavailable.")
        raise _failure from error


def extension_available() -> bool:
    """Return whether the optional CUDA extension can be loaded safely."""
    try:
        load_extension()
    except CompileError:
        return False
    return True


def extension_error() -> str | None:
    """Return the cached extension failure message, if loading has failed."""
    if _failure is None:
        return None
    return str(_failure)
