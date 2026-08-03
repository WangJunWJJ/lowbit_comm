"""Immutable plans and compile-time communication context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .config import CompressionConfig
from .stage import CommunicationStage, _require_non_empty


@dataclass(frozen=True)
class WorkspacePolicy:
    """Control optional workspace caching without owning runtime buffers."""

    cache: bool = True
    max_cached_bytes: int | None = None
    max_entries: int | None = None
    stream_safe: bool = True

    def __post_init__(self) -> None:
        if self.max_cached_bytes is not None and self.max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be >= 0")
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError("max_entries must be > 0")


@dataclass(frozen=True)
class CommunicationPlan:
    """A validated, backend-independent request for communication."""

    collective: str
    strategy: str
    backend: str = "cuda"
    compression: CompressionConfig | None = None
    stages: tuple[CommunicationStage, ...] = ()
    fallback: tuple[str, ...] = ()
    output_layout: str = "full"
    async_op: bool = True
    workspace_policy: WorkspacePolicy = WorkspacePolicy()

    def __post_init__(self) -> None:
        for field_name in ("collective", "strategy", "backend", "output_layout"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.compression is not None and not isinstance(self.compression, CompressionConfig):
            raise TypeError("compression must be a CompressionConfig or None")
        if not isinstance(self.workspace_policy, WorkspacePolicy):
            raise TypeError("workspace_policy must be a WorkspacePolicy")

        stages = tuple(self.stages)
        if any(not isinstance(stage, CommunicationStage) for stage in stages):
            raise TypeError("stages must contain only CommunicationStage values")
        fallback = tuple(self.fallback)
        for strategy in fallback:
            _require_non_empty(strategy, "fallback strategy")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "fallback", fallback)

        if self.strategy == "hierarchical" and not stages:
            raise ValueError("hierarchical strategy requires at least one stage")


@dataclass(frozen=True)
class CompileContext:
    """Static tensor, rank, topology, and resource facts used by compilation."""

    rank: int
    world_size: int
    device: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"
    local_rank: int | None = None
    local_world_size: int | None = None
    node_id: int | None = None
    node_count: int | None = None
    process_group: object | None = None
    process_groups: Mapping[str, object] = field(default_factory=dict)
    topology_signature: str = "unknown"
    device_architecture: str = "unknown"
    workspace_budget_bytes: int | None = None
    allow_dynamic_shape: bool = False

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must be >= 0 and < world_size")
        for field_name in (
            "device",
            "dtype",
            "layout",
            "topology_signature",
            "device_architecture",
        ):
            _require_non_empty(getattr(self, field_name), field_name)

        shape = tuple(self.shape)
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape):
            raise ValueError("shape dimensions must be non-negative integers")
        object.__setattr__(self, "shape", shape)

        if self.local_world_size is not None and self.local_world_size <= 0:
            raise ValueError("local_world_size must be > 0")
        if self.local_rank is not None:
            if self.local_rank < 0:
                raise ValueError("local_rank must be >= 0")
            if self.local_world_size is not None and self.local_rank >= self.local_world_size:
                raise ValueError("local_rank must be < local_world_size")
        if self.node_count is not None and self.node_count <= 0:
            raise ValueError("node_count must be > 0")
        if self.node_id is not None:
            if self.node_id < 0:
                raise ValueError("node_id must be >= 0")
            if self.node_count is not None and self.node_id >= self.node_count:
                raise ValueError("node_id must be < node_count")
        if self.workspace_budget_bytes is not None and self.workspace_budget_bytes < 0:
            raise ValueError("workspace_budget_bytes must be >= 0")

        process_groups = dict(self.process_groups)
        if any(not isinstance(name, str) or not name.strip() for name in process_groups):
            raise ValueError("process_groups keys must be non-empty strings")
        object.__setattr__(self, "process_groups", MappingProxyType(process_groups))
