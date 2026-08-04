"""Backend-neutral metadata for reduced tensor shards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ReducedShard:
    """A rank-local reduced shard and its backend-independent metadata."""

    shard: Any
    shard_index: int
    shard_numel: int
    original_shape: tuple[int, ...]
    original_numel: int
    world_size: int
    reduce: str
    padded_numel: int | None = None
    dtype: str = "auto"
    layout: str = "flat_contiguous"
    transport: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shard_index < 0:
            raise ValueError("shard_index must be >= 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.shard_index >= self.world_size:
            raise ValueError("shard_index must be < world_size")
        if self.shard_numel < 0:
            raise ValueError("shard_numel must be >= 0")
        if self.original_numel < 0:
            raise ValueError("original_numel must be >= 0")
        original_shape = tuple(self.original_shape)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in original_shape):
            raise ValueError("original_shape dimensions must be non-negative integers")
        object.__setattr__(self, "original_shape", original_shape)
        active_padded_numel = (
            self.padded_numel if self.padded_numel is not None else self.shard_numel * self.world_size
        )
        if active_padded_numel < self.original_numel:
            raise ValueError("padded_numel must be >= original_numel")
        if active_padded_numel != self.shard_numel * self.world_size:
            raise ValueError("padded_numel must equal shard_numel * world_size")
        object.__setattr__(self, "padded_numel", active_padded_numel)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

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
    def has_padding(self) -> bool:
        return self.padding_numel > 0 or int(self.padded_numel or 0) > self.original_numel

    @property
    def is_padding_only(self) -> bool:
        return self.valid_numel == 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "shard_numel": self.shard_numel,
            "shard_offset": self.shard_offset,
            "shard_end": self.shard_end,
            "valid_numel": self.valid_numel,
            "original_shape": self.original_shape,
            "original_numel": self.original_numel,
            "padded_numel": self.padded_numel,
            "world_size": self.world_size,
            "reduce": self.reduce,
            "dtype": self.dtype,
            "layout": self.layout,
            "transport": self.transport,
            "metadata": dict(self.metadata),
        }
