from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import Generic, TypeVar


T = TypeVar("T")


class WorkspaceBudgetExceeded(RuntimeError):
    """Raised when a new workspace would exceed the configured budget."""


class WorkspaceLease(Generic[T]):
    """Exclusive ownership of a pooled workspace until release."""

    def __init__(self, pool: WorkspacePool[T], key: Hashable, value: T) -> None:
        self._pool = pool
        self._key = key
        self._value = value
        self._released = False
        self._lock = RLock()

    @property
    def value(self) -> T:
        with self._lock:
            if self._released:
                raise RuntimeError("workspace lease has been released")
            return self._value

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            value = self._value
        self._pool._return(self._key, value)


class WorkspacePool(Generic[T]):
    """Thread-safe keyed pool that never reuses an active lease."""

    def __init__(self) -> None:
        self._available: dict[Hashable, list[T]] = defaultdict(list)
        self._lock = RLock()

    def acquire(self, key: Hashable, allocator: Callable[[], T]) -> WorkspaceLease[T]:
        with self._lock:
            values = self._available[key]
            value = values.pop() if values else allocator()
        return WorkspaceLease(self, key, value)

    def available(self, key: Hashable) -> int:
        with self._lock:
            return len(self._available[key])

    def _return(self, key: Hashable, value: T) -> None:
        with self._lock:
            self._available[key].append(value)


class BudgetedWorkspacePool(WorkspacePool[T]):
    """Workspace pool that accounts for retained allocations by byte size."""

    def __init__(self, budget_bytes: int | None = None) -> None:
        super().__init__()
        if budget_bytes is not None and budget_bytes < 0:
            raise ValueError("workspace budget must be >= 0")
        self._budget_bytes = budget_bytes
        self._allocated_bytes = 0

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    def acquire(
        self,
        key: Hashable,
        size_bytes: int,
        allocator: Callable[[], T],
    ) -> WorkspaceLease[T]:
        if size_bytes < 0:
            raise ValueError("workspace size must be >= 0")
        with self._lock:
            values = self._available[key]
            if values:
                value = values.pop()
            else:
                requested = self._allocated_bytes + size_bytes
                if self._budget_bytes is not None and requested > self._budget_bytes:
                    raise WorkspaceBudgetExceeded(
                        f"workspace allocation exceeds budget: "
                        f"requested={requested} budget={self._budget_bytes}"
                    )
                value = allocator()
                self._allocated_bytes = requested
        return WorkspaceLease(self, key, value)
