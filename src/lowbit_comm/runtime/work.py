from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Generic, Protocol, TypeVar, cast

from .event import CompletionEvent, ImmediateCompletionEvent


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CompletionOutcome(Generic[T]):
    """Postprocessed result plus the event that makes it safe to consume."""

    result: T
    output_ready: CompletionEvent


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
        complete: Callable[[], T | CompletionOutcome[T]] | None = None,
        resources: Iterable[Any] = (),
    ) -> None:
        self._result = result
        self._event = event or ImmediateCompletionEvent()
        self._output_ready: CompletionEvent | None = None
        self._complete = complete
        self._resources = tuple(resources)
        self._finished = False
        self._error: BaseException | None = None
        self._lock = RLock()

    def query(self) -> bool:
        with self._lock:
            if self._finished:
                return True
            if self._error is not None:
                return True
            if self._output_ready is not None:
                if not self._output_ready.query():
                    return False
                self._finish_locked()
                return True
            if not self._event.query():
                return False
            self._run_completion_locked()
            if self._output_ready is not None and not self._output_ready.query():
                return False
            self._finish_locked()
            return True

    def wait(self, timeout: float | None = None) -> T:
        deadline = monotonic() + timeout if timeout is not None else None
        if not self._event.wait(timeout):
            raise TimeoutError("communication work did not complete before timeout")

        with self._lock:
            if not self._finished and self._output_ready is None:
                self._run_completion_locked()

        if self._output_ready is not None:
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            if not self._output_ready.wait(remaining):
                raise TimeoutError("communication work did not complete before timeout")

        with self._lock:
            if not self._finished:
                self._finish_locked()

            if self._error is not None:
                raise self._error
            return cast(T, self._result)

    def _run_completion_locked(self) -> None:
        try:
            if self._complete is None:
                return
            completed = self._complete()
            if isinstance(completed, CompletionOutcome):
                self._result = completed.result
                self._output_ready = completed.output_ready
            else:
                self._result = completed
        except BaseException as error:
            self._error = error

    def _finish_locked(self) -> None:
        for resource in reversed(self._resources):
            release = getattr(resource, "release", None)
            if callable(release):
                release()
        self._resources = ()
        self._finished = True
