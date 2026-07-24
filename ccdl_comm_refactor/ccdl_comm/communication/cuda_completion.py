from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any


class NoopCompletion:
    """Completion object for runtimes that do not need CUDA event ordering."""

    def wait(self) -> None:
        return None

    def synchronize(self) -> None:
        return None


class CudaCompletion:
    """Small wrapper around a CUDA event-like object."""

    def __init__(self, event: Any | None) -> None:
        self._event = event

    def wait(self) -> None:
        if self._event is None:
            return
        wait = getattr(self._event, "wait", None)
        if callable(wait):
            wait()

    def synchronize(self) -> None:
        if self._event is None:
            return
        synchronize = getattr(self._event, "synchronize", None)
        if callable(synchronize):
            synchronize()


class CudaCompletionManager:
    """Create completion objects without making torch a hard import dependency."""

    def __init__(self, torch_provider: Callable[[], Any] | None = None) -> None:
        self._torch_provider = torch_provider or _import_torch

    def record_for(self, tensor: Any) -> CudaCompletion | NoopCompletion:
        if not bool(getattr(tensor, "is_cuda", False)):
            return NoopCompletion()
        torch = self._safe_torch()
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if cuda is None or not callable(is_available) or not is_available():
            return NoopCompletion()
        event_type = getattr(cuda, "Event", None)
        if event_type is None:
            return NoopCompletion()
        event = event_type()
        record = getattr(event, "record", None)
        if callable(record):
            record()
        return CudaCompletion(event)

    def _safe_torch(self) -> Any | None:
        try:
            return self._torch_provider()
        except (ImportError, ModuleNotFoundError):
            return None


def _import_torch() -> Any:
    return import_module("torch")
