"""Mathematical operations represented by the Semantic IR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReduceMean:
    """Compute the elementwise mean across participating ranks."""


@dataclass(frozen=True, slots=True)
class ReduceSum:
    """Compute the elementwise sum across participating ranks."""


@dataclass(frozen=True, slots=True)
class ReductionContract:
    """Compiled normalization rule shared by every execution path."""

    name: str
    world_size: int
    divisor: int

    def __post_init__(self) -> None:
        if self.name not in {"sum", "mean"}:
            raise ValueError("reduction name must be sum or mean")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        expected_divisor = 1 if self.name == "sum" else self.world_size
        if self.divisor != expected_divisor:
            raise ValueError(
                f"{self.name} reduction requires divisor {expected_divisor}"
            )


def compile_reduction(
    operation: ReduceMean | ReduceSum,
    world_size: int,
) -> ReductionContract:
    """Compile a semantic operation into one explicit normalization rule."""

    if isinstance(operation, ReduceSum):
        return ReductionContract("sum", world_size, 1)
    if isinstance(operation, ReduceMean):
        return ReductionContract("mean", world_size, world_size)
    raise TypeError(f"unsupported reduction operation {type(operation).__name__}")
