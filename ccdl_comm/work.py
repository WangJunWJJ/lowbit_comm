"""Backend-neutral asynchronous work primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .execution_info import ExecutionCounters, ExecutionInfo


T = TypeVar("T")


class CollectiveWork(Generic[T]):
    """A small async-result protocol for CCDL collective operations."""

    def wait(self) -> T:
        raise NotImplementedError

    def query(self) -> bool:
        raise NotImplementedError

    def get_future(self) -> Any | None:
        return None

    @property
    def execution_info(self) -> ExecutionInfo | None:
        """Return immutable compile-time metadata when work is compiled."""

        return None

    @property
    def execution_counters(self) -> ExecutionCounters | None:
        """Return shared lightweight counters when work is compiled."""

        return None


@dataclass(frozen=True)
class ImmediateWork(CollectiveWork[T]):
    """A completed collective result exposed through the async work API."""

    result: T
    execution_info: ExecutionInfo | None = field(default=None, compare=False, repr=False)
    execution_counters: ExecutionCounters | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.execution_counters is not None:
            self.execution_counters._record_completed()

    def wait(self) -> T:
        if self.execution_counters is not None:
            self.execution_counters._record_wait()
        return self.result

    def query(self) -> bool:
        if self.execution_counters is not None:
            self.execution_counters._record_query()
        return True


class CompletionWork(CollectiveWork[T]):
    """Own an asynchronous handle, deferred completion, and in-flight buffers."""

    def __init__(
        self,
        result: T,
        *,
        handle: Any | None = None,
        complete: Callable[[], T] | None = None,
        completion: Any | None = None,
        resources: Sequence[Any] = (),
        execution_info: ExecutionInfo | None = None,
        execution_counters: ExecutionCounters | None = None,
    ) -> None:
        self._result = result
        self._handle = handle
        self._complete = complete
        self._completion = completion
        self._resources = tuple(resources)
        self._finished = False
        self._callback_finished = complete is None
        self._error: BaseException | None = None
        self._execution_info = execution_info
        self._execution_counters = execution_counters
        self._terminal_recorded = False

    @property
    def resources(self) -> tuple[Any, ...]:
        return self._resources

    @property
    def execution_info(self) -> ExecutionInfo | None:
        return self._execution_info

    @property
    def execution_counters(self) -> ExecutionCounters | None:
        return self._execution_counters

    def _bind_execution(
        self,
        execution_info: ExecutionInfo,
        execution_counters: ExecutionCounters,
    ) -> None:
        self._execution_info = execution_info
        if self._execution_counters is not None and self._execution_counters is not execution_counters:
            raise ValueError("work already has different execution_counters")
        self._execution_counters = execution_counters
        if self._finished:
            self._record_terminal()

    def wait(self) -> T:
        if self._execution_counters is not None:
            self._execution_counters._record_wait()
        if not self._finished:
            try:
                self._wait_handle()
                if not self._callback_finished and self._complete is not None:
                    self._result = self._complete()
                    self._callback_finished = True
                self._wait_completion()
            except BaseException as exc:
                self._error = exc
            finally:
                self._finished = True
                self._record_terminal()
        if self._error is not None:
            raise self._error
        return self._result

    def query(self) -> bool:
        if self._execution_counters is not None:
            self._execution_counters._record_query()
        if self._finished:
            return True
        if not self._query_object(self._handle):
            return False
        if not self._callback_finished:
            return False
        return self._query_object(self._completion)

    def _record_terminal(self) -> None:
        if self._terminal_recorded or self._execution_counters is None:
            return
        if self._error is None:
            self._execution_counters._record_completed()
        else:
            self._execution_counters._record_failed()
        self._terminal_recorded = True

    def get_future(self) -> Any | None:
        get_future = getattr(self._handle, "get_future", None)
        if callable(get_future):
            return get_future()
        return None

    def _wait_handle(self) -> None:
        wait = getattr(self._handle, "wait", None)
        if callable(wait):
            wait()

    def _wait_completion(self) -> None:
        wait = getattr(self._completion, "wait", None)
        if callable(wait):
            wait()

    @staticmethod
    def _query_object(value: Any | None) -> bool:
        if value is None:
            return True
        for name in ("is_completed", "query"):
            query = getattr(value, name, None)
            if callable(query):
                return bool(query())
        return False


class BoundCollectiveWork(CollectiveWork[T]):
    """Attach compiled metadata to a foreign work-compatible object."""

    __slots__ = (
        "_delegate",
        "_execution_info",
        "_execution_counters",
        "_terminal_recorded",
        "_finished",
        "_result",
        "_error",
    )

    def __init__(
        self,
        delegate: Any,
        execution_info: ExecutionInfo,
        execution_counters: ExecutionCounters,
    ) -> None:
        self._delegate = delegate
        self._execution_info = execution_info
        self._execution_counters = execution_counters
        self._terminal_recorded = False
        self._finished = False
        self._result = None
        self._error: BaseException | None = None

    @property
    def execution_info(self) -> ExecutionInfo:
        return self._execution_info

    @property
    def execution_counters(self) -> ExecutionCounters:
        return self._execution_counters

    @property
    def resources(self) -> tuple[Any, ...]:
        return tuple(getattr(self._delegate, "resources", ()))

    def wait(self) -> T:
        self._execution_counters._record_wait()
        if not self._finished:
            try:
                self._result = self._delegate.wait()
            except BaseException as exc:
                self._error = exc
            finally:
                self._finished = True
                self._record_terminal(failed=self._error is not None)
        if self._error is not None:
            raise self._error
        return self._result

    def query(self) -> bool:
        self._execution_counters._record_query()
        if self._finished:
            return True
        query = getattr(self._delegate, "query", None)
        if callable(query):
            return bool(query())
        is_completed = getattr(self._delegate, "is_completed", None)
        return bool(is_completed()) if callable(is_completed) else False

    def get_future(self) -> Any | None:
        get_future = getattr(self._delegate, "get_future", None)
        return get_future() if callable(get_future) else None

    def _record_terminal(self, *, failed: bool) -> None:
        if self._terminal_recorded:
            return
        if failed:
            self._execution_counters._record_failed()
        else:
            self._execution_counters._record_completed()
        self._terminal_recorded = True


def bind_execution_work(
    result: Any,
    execution_info: ExecutionInfo,
    execution_counters: ExecutionCounters,
) -> CollectiveWork[Any]:
    """Attach one executor's immutable metadata and counters to a result."""

    if isinstance(result, ImmediateWork):
        if result.execution_counters is execution_counters:
            object.__setattr__(result, "execution_info", execution_info)
            return result
        return ImmediateWork(
            result.result,
            execution_info=execution_info,
            execution_counters=execution_counters,
        )
    if isinstance(result, CompletionWork):
        result._bind_execution(execution_info, execution_counters)
        return result
    if isinstance(result, CollectiveWork):
        return BoundCollectiveWork(result, execution_info, execution_counters)
    has_wait = callable(getattr(result, "wait", None))
    has_query = callable(getattr(result, "query", None)) or callable(
        getattr(result, "is_completed", None)
    )
    if has_wait and has_query:
        return BoundCollectiveWork(result, execution_info, execution_counters)
    return ImmediateWork(
        result,
        execution_info=execution_info,
        execution_counters=execution_counters,
    )
