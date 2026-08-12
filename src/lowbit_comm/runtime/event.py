from __future__ import annotations

from threading import Event
from typing import Protocol


class CompletionEvent(Protocol):
    """Signals that asynchronous backend work is safe to consume."""

    def query(self) -> bool:
        """Return whether the event has completed without blocking."""

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for completion and return whether completion was observed."""


class ManualCompletionEvent:
    """A backend-neutral completion event used by tests and CPU backends."""

    def __init__(self, *, completed: bool = False) -> None:
        self._event = Event()
        if completed:
            self._event.set()

    def complete(self) -> None:
        self._event.set()

    def query(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class ImmediateCompletionEvent(ManualCompletionEvent):
    """An event that is complete at construction time."""

    def __init__(self) -> None:
        super().__init__(completed=True)
