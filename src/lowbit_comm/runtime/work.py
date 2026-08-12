from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock
from typing import Any, Generic, Protocol, TypeVar, cast

from .event import CompletionEvent, ImmediateCompletionEvent


T = TypeVar("T")


class Releasable(Protocol):
    def release(self) -> None: ...


class Work(Protocol, Generic[T]):
    def query(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> T: ...


class CompletionWork(Generic[T]):
    """Owns postprocessing and resources until the completion event fires."""

    def __init__(
        self,
        result: T,
        *,
        event: CompletionEvent | None = None,
        complete: Callable[[], T] | None = None,
        resources: Iterable[Any] = (),
    ) -> None:
        self._result = result
        self._event = event or ImmediateCompletionEvent()
        self._complete = complete
        self._resources = tuple(resources)
        self._finished = False
        self._error: BaseException | None = None
        self._lock = RLock()

    def query(self) -> bool:
        with self._lock:
            return self._finished

    def wait(self, timeout: float | None = None) -> T:
        if not self._event.wait(timeout):
            raise TimeoutError("communication work did not complete before timeout")

        with self._lock:
            if not self._finished:
                try:
                    if self._complete is not None:
                        self._result = self._complete()
                except BaseException as error:
                    self._error = error
                finally:
                    for resource in reversed(self._resources):
                        release = getattr(resource, "release", None)
                        if callable(release):
                            release()
                    self._finished = True

            if self._error is not None:
                raise self._error
            return cast(T, self._result)
