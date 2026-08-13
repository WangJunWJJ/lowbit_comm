"""Ordered completion pipeline for transport, GPU stages, and resources."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any, Generic, TypeVar, cast

from .event import CompletionEvent


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CompletionStage(Generic[T]):
    name: str
    event: CompletionEvent
    action: Callable[[T], T] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("completion stage name must be non-empty")


class CompletionPipeline(Generic[T]):
    """Advance an immutable sequence of already-scheduled completion stages."""

    def __init__(self, result: T, *, resources: Iterable[Any] = ()) -> None:
        self._result = result
        self._stages: list[CompletionStage[T]] = []
        self._resources = tuple(resources)
        self._next_stage = 0
        self._submitted = False
        self._finished = False
        self._error: BaseException | None = None
        self._futures: list[Any] = []
        self._lock = RLock()

    def add_stage(
        self,
        name: str,
        event: CompletionEvent,
        *,
        action: Callable[[T], T] | None = None,
    ) -> None:
        with self._lock:
            if self._submitted:
                raise RuntimeError("completion pipeline is already submitted")
            self._stages.append(CompletionStage(name, event, action))

    def get_future(
        self,
        future_factory: Any,
        *,
        manager: CompletionManager | None = None,
    ) -> Any:
        future = future_factory()
        with self._lock:
            self._submitted = True
            if self._finished:
                self._resolve_future_locked(future)
            else:
                self._futures.append(future)
        if manager is not None:
            manager.submit(self)
        return future

    def query(self) -> bool:
        with self._lock:
            self._submitted = True
            self._advance_ready_locked()
            return self._finished

    def wait(self, timeout: float | None = None) -> T:
        deadline = monotonic() + timeout if timeout is not None else None
        with self._lock:
            self._submitted = True
        while True:
            with self._lock:
                self._advance_ready_locked()
                if self._finished:
                    return self._result_or_raise_locked()
                stage = self._stages[self._next_stage]
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            if not stage.event.wait(remaining):
                raise TimeoutError(
                    f"completion stage {stage.name!r} did not finish before timeout"
                )

    def _advance_ready_locked(self) -> None:
        if self._finished:
            return
        try:
            while self._next_stage < len(self._stages):
                stage = self._stages[self._next_stage]
                if not stage.event.query():
                    return
                if stage.action is not None:
                    self._result = stage.action(self._result)
                self._next_stage += 1
            self._finish_locked()
        except BaseException as error:
            self._error = error
            self._finish_locked()

    def _finish_locked(self) -> None:
        if self._finished:
            return
        for resource in reversed(self._resources):
            release = getattr(resource, "release", None)
            if callable(release):
                release()
        self._resources = ()
        self._stages.clear()
        self._finished = True
        for future in self._futures:
            self._resolve_future_locked(future)
        self._futures.clear()

    def _resolve_future_locked(self, future: Any) -> None:
        if self._error is None:
            future.set_result(self._result)
        else:
            future.set_exception(self._error)

    def _result_or_raise_locked(self) -> T:
        if self._error is not None:
            raise self._error
        return cast(T, self._result)


class CompletionManager:
    """Advance many pipelines from one shared completion thread."""

    def __init__(self, *, poll_interval: float = 0.001) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._poll_interval = poll_interval
        self._pipelines: set[CompletionPipeline[Any]] = set()
        self._condition = Condition()
        self._closed = False
        self._thread = Thread(
            target=self._run,
            name="lowbit-completion-manager",
            daemon=True,
        )
        self._thread.start()

    def submit(self, pipeline: CompletionPipeline[Any]) -> None:
        if not isinstance(pipeline, CompletionPipeline):
            raise TypeError("completion manager requires a CompletionPipeline")
        with self._condition:
            if self._closed:
                raise RuntimeError("completion manager is closed")
            self._pipelines.add(pipeline)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pipelines and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pipelines:
                    return
                pipelines = tuple(self._pipelines)
            completed = tuple(pipeline for pipeline in pipelines if pipeline.query())
            with self._condition:
                self._pipelines.difference_update(completed)
                if self._closed and not self._pipelines:
                    return
                self._condition.wait(self._poll_interval)
