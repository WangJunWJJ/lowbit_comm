"""Immutable flat parameter layout for ReducedShard consumers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from ccdl_comm.shard import ReducedShard


@dataclass(frozen=True, slots=True)
class FlatParameterSlice:
    """One parameter's location in a deterministic flattened model layout."""

    index: int
    offset: int
    numel: int
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.index, "index")
        _require_nonnegative_integer(self.offset, "offset")
        _require_nonnegative_integer(self.numel, "numel")
        shape = tuple(self.shape)
        for dimension in shape:
            _require_nonnegative_integer(dimension, "shape dimension")
        if reduce(mul, shape, 1) != self.numel:
            raise ValueError("shape product must equal numel")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise TypeError("dtype must be a non-empty string")
        if not isinstance(self.requires_grad, bool):
            raise TypeError("requires_grad must be a boolean")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class FlatShardLayout:
    """Model-wide flat layout and the rank-local range owned by one consumer."""

    original_numel: int
    padded_numel: int
    shard_numel: int
    world_size: int
    shard_index: int
    parameters: tuple[FlatParameterSlice, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.original_numel, "original_numel")
        _require_nonnegative_integer(self.padded_numel, "padded_numel")
        _require_nonnegative_integer(self.shard_numel, "shard_numel")
        _require_positive_integer(self.world_size, "world_size")
        _require_nonnegative_integer(self.shard_index, "shard_index")
        if self.shard_index >= self.world_size:
            raise ValueError("shard_index must be smaller than world_size")
        if self.padded_numel != self.shard_numel * self.world_size:
            raise ValueError("padded_numel must equal shard_numel * world_size")
        if self.padded_numel < self.original_numel:
            raise ValueError("padded_numel must be >= original_numel")

        parameters = tuple(self.parameters)
        cursor = 0
        dtypes: set[str] = set()
        for expected_index, parameter in enumerate(parameters):
            if parameter.index != expected_index or parameter.offset != cursor:
                raise ValueError(
                    "parameter slices must be contiguous through original_numel"
                )
            cursor += parameter.numel
            dtypes.add(parameter.dtype)
        if cursor != self.original_numel:
            raise ValueError(
                "parameter slices must be contiguous through original_numel"
            )
        if len(dtypes) > 1:
            raise ValueError("parameter slices must use one dtype")
        object.__setattr__(self, "parameters", parameters)

    @property
    def shard_offset(self) -> int:
        return self.shard_index * self.shard_numel

    @property
    def shard_end(self) -> int:
        return min(self.shard_offset + self.shard_numel, self.original_numel)

    @property
    def valid_numel(self) -> int:
        return max(0, self.shard_end - self.shard_offset)

    @property
    def padding_numel(self) -> int:
        return self.shard_numel - self.valid_numel

    @property
    def logical_range(self) -> tuple[int, int]:
        return self.shard_offset, self.shard_end

    @property
    def dtype(self) -> str:
        return self.parameters[0].dtype if self.parameters else "auto"

    def validate_reduced_shard(self, reduced: ReducedShard) -> None:
        """Reject a shard that cannot safely update this layout."""

        expected = {
            "shard_index": self.shard_index,
            "world_size": self.world_size,
            "shard_numel": self.shard_numel,
            "original_numel": self.original_numel,
            "padded_numel": self.padded_numel,
            "dtype": self.dtype,
        }
        for name, value in expected.items():
            if getattr(reduced, name) != value:
                raise ValueError(
                    f"ReducedShard {name} does not match layout: "
                    f"{getattr(reduced, name)!r} != {value!r}"
                )
        if tuple(reduced.original_shape) != (self.original_numel,):
            raise ValueError("ReducedShard original_shape must describe the flat layout")
        if reduced.logical_range != self.logical_range:
            raise ValueError("ReducedShard logical range does not match layout")
        numel = getattr(reduced.shard, "numel", None)
        if not callable(numel) or int(numel()) != self.shard_numel:
            raise ValueError("ReducedShard tensor numel must equal shard_numel")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_integer(value: object, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be > 0")
