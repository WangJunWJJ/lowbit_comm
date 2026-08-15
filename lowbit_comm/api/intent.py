"""Immutable communication intent contracts."""

from dataclasses import dataclass
from enum import Enum
from math import prod

from lowbit_comm.core.errors import CompileError


class ReductionOp(str, Enum):
    """Reduction operations supported by a communication intent."""

    SUM = "sum"
    MEAN = "mean"


class OutputSemantics(str, Enum):
    """The result layout requested by a communication intent."""

    FULL_TENSOR = "full_tensor"
    REDUCED_SHARD = "reduced_shard"


class CompletionMode(str, Enum):
    """The completion behavior requested by a communication intent."""

    SYNC = "sync"
    ASYNC = "async"


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Description of a tensor participating in communication."""

    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.dtype) is not str:
            raise CompileError("Tensor dtype must be a string.")
        if not _is_valid_shape(self.shape):
            raise CompileError(
                "Tensor shape must be non-empty and non-negative."
            )

    @property
    def numel(self) -> int:
        """Return the number of tensor elements."""
        return prod(self.shape)


@dataclass(frozen=True, slots=True)
class ShapeFamily:
    """Bounds and alignment constraints for a tensor shape family."""

    max_numel: int
    alignment: int

    def __post_init__(self) -> None:
        if type(self.max_numel) is not int:
            raise CompileError("Shape-family maximum must be an integer.")
        if type(self.alignment) is not int:
            raise CompileError("Shape-family alignment must be an integer.")
        if self.max_numel < 0:
            raise CompileError("Shape-family maximum must be non-negative.")
        if self.alignment <= 0:
            raise CompileError("Shape-family alignment must be positive.")

    def accepts(self, tensor: TensorSpec) -> bool:
        """Return whether *tensor* belongs to this shape family."""
        return (
            tensor.numel <= self.max_numel
            and tensor.numel % self.alignment == 0
        )


@dataclass(frozen=True, slots=True)
class CommunicationIntent:
    """Fully specified, backend-independent communication request."""

    tensor: TensorSpec
    shape_family: ShapeFamily
    reduction: ReductionOp
    output: OutputSemantics
    completion: CompletionMode
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        _validate_intent_types(self)
        if self.world_size <= 0:
            raise CompileError("World size must be positive.")
        if not 0 <= self.rank < self.world_size:
            raise CompileError("Rank must belong to the world-size domain.")
        if not self.shape_family.accepts(self.tensor):
            raise CompileError("Tensor does not belong to the shape family.")


def _is_valid_shape(shape: object) -> bool:
    """Return whether *shape* is a non-empty tuple of non-negative integers."""
    return (
        type(shape) is tuple
        and bool(shape)
        and all(
            type(dimension) is int
            and dimension >= 0
            for dimension in shape
        )
    )


def _validate_intent_types(intent: CommunicationIntent) -> None:
    """Reject non-exact values from the immutable intent signature."""
    if type(intent.tensor) is not TensorSpec:
        raise CompileError("Intent tensor must be a TensorSpec.")
    if type(intent.shape_family) is not ShapeFamily:
        raise CompileError("Intent shape family must be a ShapeFamily.")
    if type(intent.reduction) is not ReductionOp:
        raise CompileError("Intent reduction must be a ReductionOp.")
    if type(intent.output) is not OutputSemantics:
        raise CompileError("Intent output must be OutputSemantics.")
    if type(intent.completion) is not CompletionMode:
        raise CompileError("Intent completion must be CompletionMode.")
    if type(intent.world_size) is not int:
        raise CompileError("World size must be an integer.")
    if type(intent.rank) is not int:
        raise CompileError("Rank must be an integer.")
