"""Backend-neutral runtime values returned by compiled executables."""

from __future__ import annotations

from dataclasses import dataclass

from .types import DataType


@dataclass(frozen=True, slots=True)
class ReducedShardValue:
    """One globally reduced logical shard and immutable layout metadata."""

    tensor: object
    shard_index: int
    shard_numel: int
    original_shape: tuple[int, ...]
    original_numel: int
    world_size: int
    reduction: str
    dtype: DataType
    layout_version: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.shard_index < 0 or self.shard_index >= self.world_size:
            raise ValueError("shard_index must be within world_size")
        if self.shard_numel < 0 or self.original_numel < 0:
            raise ValueError("shard and original sizes must be non-negative")
        if self.shard_numel * self.world_size < self.original_numel:
            raise ValueError("shard layout cannot represent the original tensor")
        object.__setattr__(self, "original_shape", tuple(self.original_shape))
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be mean or sum")
        if not isinstance(self.dtype, DataType):
            raise TypeError("dtype must be a DataType")

    @property
    def shard_offset(self) -> int:
        return self.shard_index * self.shard_numel

    @property
    def shard_end(self) -> int:
        return min(self.shard_offset + self.shard_numel, self.original_numel)

    @property
    def logical_range(self) -> tuple[int, int]:
        start = min(self.shard_offset, self.original_numel)
        return start, self.shard_end

    @property
    def valid_numel(self) -> int:
        return max(0, self.shard_end - self.shard_offset)

    @property
    def padding_numel(self) -> int:
        return self.shard_numel - self.valid_numel
