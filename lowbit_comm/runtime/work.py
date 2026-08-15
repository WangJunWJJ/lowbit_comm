"""Generic completion contracts for communication execution."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from lowbit_comm.core.errors import ExecutionError


T = TypeVar("T")


class CommunicationWork(Protocol, Generic[T]):
    """Structural completion protocol implemented by runtime backends."""

    def is_completed(self) -> bool:
        """Return whether execution reached a terminal state."""
        ...

    def wait(self) -> T:
        """Wait for completion and return the published result."""
        ...

    def result(self) -> T:
        """Return the result or propagate the execution failure."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class CompletedWork(Generic[T]):
    """Already-completed work that publishes one result."""

    value: T

    def is_completed(self) -> bool:
        """Return true because the result is already available."""
        return True

    def wait(self) -> T:
        """Return the completed result without synchronization."""
        return self.value

    def result(self) -> T:
        """Return the completed result without synchronization."""
        return self.value


@dataclass(frozen=True, slots=True, eq=False)
class FailedWork(Generic[T]):
    """Already-failed work that never publishes a result."""

    failure: ExecutionError

    def __post_init__(self) -> None:
        if not isinstance(self.failure, ExecutionError):
            raise TypeError(
                "FailedWork failure must be an ExecutionError."
            )

    def is_completed(self) -> bool:
        """Return true because the failure is terminal."""
        return True

    def wait(self) -> T:
        """Propagate the original execution failure."""
        raise self.failure

    def result(self) -> T:
        """Propagate the original execution failure."""
        raise self.failure
