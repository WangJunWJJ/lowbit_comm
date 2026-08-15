"""Transactional publication of error-feedback residuals."""

from enum import Enum
from typing import Generic, TypeVar

from lowbit_comm.core.errors import ExecutionError


T = TypeVar("T")


class FeedbackState(str, Enum):
    """Lifecycle state of an error-feedback transaction."""

    CREATED = "created"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"


class ErrorFeedbackTransaction(Generic[T]):
    """Publish a candidate residual only after successful completion."""

    __slots__ = (
        "_abort_reason",
        "_candidate",
        "_previous",
        "_state",
    )

    def __init__(self, *, previous: T, candidate: T) -> None:
        self._previous = previous
        self._candidate = candidate
        self._state = FeedbackState.CREATED
        self._abort_reason: ExecutionError | None = None

    @property
    def state(self) -> FeedbackState:
        """Return the current transaction state."""
        return self._state

    @property
    def visible_residual(self) -> T:
        """Return the residual published by the transaction."""
        if self._state is FeedbackState.COMMITTED:
            return self._candidate
        return self._previous

    def prepare(self) -> None:
        """Mark the candidate ready without publishing it."""
        if self._state is FeedbackState.CREATED:
            self._state = FeedbackState.PREPARED
            return
        if self._state is FeedbackState.PREPARED:
            raise ExecutionError("Error feedback is already prepared.")
        if self._state is FeedbackState.ABORTED:
            error = ExecutionError(
                "Cannot prepare aborted error feedback."
            )
            raise error from self._abort_reason
        raise ExecutionError(
            f"Cannot prepare {self._state.value} error feedback."
        )

    def commit(self) -> T:
        """Publish and return a successfully prepared candidate."""
        if self._state is FeedbackState.PREPARED:
            self._state = FeedbackState.COMMITTED
            return self._candidate
        if self._state is FeedbackState.CREATED:
            raise ExecutionError(
                "Cannot commit error feedback before prepare."
            )
        if self._state is FeedbackState.COMMITTED:
            raise ExecutionError("Error feedback is already committed.")
        error = ExecutionError("Cannot commit aborted error feedback.")
        raise error from self._abort_reason

    def abort(self, reason: ExecutionError) -> None:
        """Discard the candidate and retain the execution failure."""
        if not isinstance(reason, ExecutionError):
            raise TypeError("Abort reason must be an ExecutionError.")
        if self._state in (
            FeedbackState.CREATED,
            FeedbackState.PREPARED,
        ):
            self._abort_reason = reason
            self._state = FeedbackState.ABORTED
            return
        if self._state is FeedbackState.ABORTED:
            error = ExecutionError("Error feedback is already aborted.")
            raise error from self._abort_reason
        raise ExecutionError("Cannot abort committed error feedback.")
