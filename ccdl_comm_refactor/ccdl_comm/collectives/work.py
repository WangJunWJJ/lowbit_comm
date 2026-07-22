from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


class CollectiveWork(Generic[T]):
    """A small async-result protocol for CCDL collective operations."""

    def wait(self) -> T:
        """Block until the collective has completed and return its result."""

        raise NotImplementedError


@dataclass(frozen=True)
class ImmediateWork(CollectiveWork[T]):
    """A completed collective result exposed through the async work API."""

    result: T

    def wait(self) -> T:
        """Return the already-computed collective result."""

        return self.result
