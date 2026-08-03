from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from importlib import import_module
from typing import Any

from ccdl_comm.collectives.work import CompletionWork
from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.cuda.loader import CudaExtensionStatus, load_cuda_extension


NATIVE_WORK_ABI_VERSION = 1


class NoopCompletion:
    """Completion object for runtimes that do not need CUDA event ordering."""

    def wait(self) -> None:
        return None

    def wait_stream(self, stream: Any) -> None:
        del stream
        return None

    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True


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

    def wait_stream(self, stream: Any) -> None:
        if self._event is None:
            return
        wait = getattr(self._event, "wait", None)
        if callable(wait):
            wait(stream)

    def synchronize(self) -> None:
        if self._event is None:
            return
        synchronize = getattr(self._event, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def query(self) -> bool:
        if self._event is None:
            return True
        query = getattr(self._event, "query", None)
        if callable(query):
            return bool(query())
        return False


class CudaStreamWork:
    """Context-managed CUDA stream/event work compatible with distributed handles."""

    def __init__(self, *, torch: Any, async_op: bool = False, handle: Any | None = None) -> None:
        self.async_op = async_op
        self.handle = handle
        self._torch = torch
        cuda = getattr(torch, "cuda")
        self.stream = _comm_stream_for(cuda) if async_op else cuda.current_stream()
        self._stream_guard = cuda.stream(self.stream)
        self.event = None

    def __enter__(self) -> CudaStreamWork:
        cuda = getattr(self._torch, "cuda")
        self.stream.wait_stream(cuda.current_stream())
        self._stream_guard.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stream_guard.__exit__(exc_type, exc_value, traceback)
        event_type = getattr(getattr(self._torch, "cuda"), "Event")
        self.event = event_type()
        record = getattr(self.event, "record", None)
        if callable(record):
            record(self.stream)
        if not self.async_op:
            self._wait_event()

    def wait(self) -> None:
        wait = getattr(self.handle, "wait", None)
        if callable(wait):
            wait()
        if self.async_op:
            self._wait_event()

    def query(self) -> bool:
        if not self.async_op:
            return True
        query = getattr(self.event, "query", None)
        if callable(query):
            return bool(query())
        return False

    def _wait_event(self) -> None:
        wait = getattr(self.event, "wait", None)
        if callable(wait):
            wait()


_CUDA_COMM_STREAMS: dict[int, Any] = {}


def _comm_stream_for(cuda: Any) -> Any:
    current_device = int(cuda.current_device())
    stream = _CUDA_COMM_STREAMS.get(current_device)
    if stream is None:
        stream = cuda.Stream(current_device)
        _CUDA_COMM_STREAMS[current_device] = stream
    return stream


@lru_cache(maxsize=8)
def _native_executor_for(module: Any) -> Any | None:
    """Return one stateless native executor per loaded extension module."""

    if getattr(module, "NATIVE_WORK_ABI_VERSION", None) != NATIVE_WORK_ABI_VERSION:
        return None
    if not hasattr(module, "CompressedWork"):
        return None
    factory = getattr(module, "create_cuda_executor", None)
    if not callable(factory):
        return None
    try:
        return factory()
    except (ImportError, RuntimeError, TypeError):
        return None


def native_work_available(extension_status: CudaExtensionStatus) -> bool:
    """Return whether a compatible native Work executor can be constructed."""

    module = extension_status.module
    return bool(
        extension_status.available
        and module is not None
        and _native_executor_for(module) is not None
    )


@lru_cache(maxsize=32)
def _is_native_runtime_type(runtime_type: type) -> bool:
    module_name = runtime_type.__module__
    if module_name.startswith("torch."):
        return True
    return module_name.startswith("ccdl_comm.") and runtime_type.__name__ in {
        "AsyncAllGatherWork",
        "AsyncAllReduceWork",
        "CudaStreamWork",
    }


def _is_native_runtime_object(value: Any | None) -> bool:
    return value is not None and _is_native_runtime_type(type(value))


class CudaCompletionManager:
    """Create completion objects without making torch a hard import dependency."""

    def __init__(
        self,
        torch_provider: Callable[[], Any] | None = None,
        extension_status: CudaExtensionStatus | None = None,
    ) -> None:
        self._torch_provider = torch_provider or _import_torch
        self._extension_status = extension_status or load_cuda_extension()
        self._native_executor = self._create_native_executor()

    def record_for(self, tensor: Any, *, stream: Any | None = None) -> CudaCompletion | NoopCompletion:
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
            if stream is None:
                record()
            else:
                record(stream)
        return CudaCompletion(event)

    def create_stream_work(self, *, async_op: bool = False, handle: Any | None = None) -> CudaStreamWork | NoopCompletion:
        torch = self._safe_torch()
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if cuda is None or not callable(is_available) or not is_available():
            return NoopCompletion()
        required = ("current_device", "current_stream", "stream", "Event")
        if any(not hasattr(cuda, attr) for attr in required):
            return NoopCompletion()
        return CudaStreamWork(torch=torch, async_op=async_op, handle=handle)

    def create_work(
        self,
        *,
        result: Any,
        handle: Any | None = None,
        complete: Callable[[], Any] | None = None,
        completion: Any | None = None,
        resources: tuple[Any, ...] = (),
        execution_info: ExecutionInfo | None = None,
        execution_counters: ExecutionCounters | None = None,
    ) -> Any:
        """Create a result-bearing work object without requiring CUDA."""

        if (
            self._native_executor is not None
            and execution_info is None
            and execution_counters is None
            and (
                complete is not None
                or _is_native_runtime_object(handle)
                or _is_native_runtime_object(completion)
            )
        ):
            return self._native_executor.run(
                result,
                handle,
                completion,
                list(resources),
                complete,
            )

        fallback_info = execution_info
        if fallback_info is not None:
            fallback_info = replace(fallback_info, fast_path="python_fallback")

        return CompletionWork(
            result,
            handle=handle,
            complete=complete,
            completion=completion,
            resources=resources,
            execution_info=fallback_info,
            execution_counters=execution_counters,
        )

    def _create_native_executor(self) -> Any | None:
        module = self._extension_status.module
        if not self._extension_status.available or module is None:
            return None
        return _native_executor_for(module)

    def _safe_torch(self) -> Any | None:
        try:
            return self._torch_provider()
        except (ImportError, ModuleNotFoundError):
            return None


def _import_torch() -> Any:
    return import_module("torch")
