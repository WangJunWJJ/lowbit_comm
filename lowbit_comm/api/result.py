"""Immutable communication result contracts."""

from dataclasses import dataclass
from math import prod
from typing import Generic, TypeVar

from lowbit_comm.core.errors import CompileError


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FullTensorResult(Generic[T]):
    """A communication result that contains a full tensor value."""

    value: T


@dataclass(frozen=True, slots=True)
class ReducedShardMetadata:
    """Exact ownership information for a reduced tensor shard."""

    global_shape: tuple[int, ...]
    offset: int
    valid_length: int
    padded_length: int
    owner_rank: int

    def __post_init__(self) -> None:
        if not _is_valid_shape(self.global_shape):
            raise CompileError(
                "Global shape must be non-empty and non-negative."
            )
        if self.offset < 0:
            raise CompileError("Shard offset must be non-negative.")
        if self.valid_length < 0:
            raise CompileError("Shard valid length must be non-negative.")
        if self.padded_length < self.valid_length:
            raise CompileError(
                "Shard padded length cannot be shorter than valid length."
            )
        if self.owner_rank < 0:
            raise CompileError("Shard owner rank must be non-negative.")
        if self.stop > prod(self.global_shape):
            raise CompileError(
                "Shard ownership range exceeds the global tensor."
            )

    @property
    def stop(self) -> int:
        """Return the exclusive global index owned by this shard."""
        return self.offset + self.valid_length


@dataclass(frozen=True, slots=True)
class ReducedShardResult(Generic[T]):
    """A communication result containing one reduced tensor shard."""

    value: T
    metadata: ReducedShardMetadata


def _is_valid_shape(shape: object) -> bool:
    """Return whether *shape* is a non-empty tuple of non-negative integers."""
    return (
        isinstance(shape, tuple)
        and bool(shape)
        and all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension >= 0
            for dimension in shape
        )
    )
