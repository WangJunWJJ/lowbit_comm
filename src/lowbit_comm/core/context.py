"""Static compilation facts and separately owned runtime bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import DataType


@dataclass(frozen=True, slots=True)
class CompileContext:
    rank: int
    world_size: int
    shape: tuple[int, ...]
    dtype: DataType
    device_type: str
    device_architecture: str = "unknown"
    topology_signature: str = "unknown"
    layout_generation: int = 0
    workspace_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("world_size must be an integer")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must be within world_size")
        shape = tuple(self.shape)
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
            raise ValueError("shape dimensions must be non-negative integers")
        object.__setattr__(self, "shape", shape)
        if not isinstance(self.dtype, DataType):
            raise TypeError("dtype must be a DataType")
        for name in ("device_type", "device_architecture", "topology_signature"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.layout_generation < 0:
            raise ValueError("layout_generation must be >= 0")
        if self.workspace_budget_bytes is not None and self.workspace_budget_bytes < 0:
            raise ValueError("workspace_budget_bytes must be >= 0")


@dataclass(slots=True)
class RuntimeBindings:
    process_group: Any = None
    backend_runtime: Any = None
    stream_provider: Any = None
    allocator: Any = None
