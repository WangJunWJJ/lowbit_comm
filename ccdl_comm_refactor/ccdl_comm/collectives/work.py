from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class CollectiveWork(Generic[T]):
    """A small async-result protocol for CCDL collective operations."""

    def wait(self) -> T:
        """Block until the collective has completed and return its result."""

        raise NotImplementedError

    def query(self) -> bool:
        """Return whether the result can be consumed without blocking."""

        raise NotImplementedError

    def get_future(self) -> Any | None:
        """Return the backend future when the transport exposes one."""

        return None


@dataclass(frozen=True)
class ImmediateWork(CollectiveWork[T]):
    """A completed collective result exposed through the async work API."""

    result: T

    def wait(self) -> T:
        """Return the already-computed collective result."""

        return self.result

    def query(self) -> bool:
        """Report that the result is already available."""

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
    ) -> None:
        self._result = result
        self._handle = handle
        self._complete = complete
        self._completion = completion
        self._resources = tuple(resources)
        self._finished = False
        self._callback_finished = complete is None
        self._error: BaseException | None = None

    @property
    def resources(self) -> tuple[Any, ...]:
        """Return resources retained for the lifetime of this work."""

        return self._resources

    def wait(self) -> T:
        """Complete the backend work and deferred post-processing once."""

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
        if self._error is not None:
            raise self._error
        return self._result

    def query(self) -> bool:
        """Observe readiness without waiting or running deferred callbacks."""

        if self._finished:
            return True
        if not self._query_object(self._handle):
            return False
        if not self._callback_finished:
            return False
        return self._query_object(self._completion)

    def get_future(self) -> Any | None:
        """Return a future exposed by the distributed backend handle."""

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
