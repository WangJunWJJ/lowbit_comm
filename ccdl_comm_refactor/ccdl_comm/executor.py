"""Backend-neutral compiled executor protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import CompressionConfig
from .execution_info import ExecutionInfo
from .plan import WorkspacePolicy
from .work import CollectiveWork


@runtime_checkable
class CompiledExecutor(Protocol):
    """Execute an already-resolved communication plan on the data path."""

    execution_info: ExecutionInfo

    def run(self, tensor: object) -> CollectiveWork[object]:
        """Execute communication without repeating control-plane resolution."""

        ...


class ObjectIdentity:
    """Hash an object by identity while retaining it for the key lifetime."""

    __slots__ = ("_object", "_hash")

    def __init__(self, value: object | None) -> None:
        self._object = value
        self._hash = id(value)

    @property
    def value(self) -> object | None:
        return self._object

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObjectIdentity) and self._object is other._object


@dataclass(frozen=True)
class CompileCacheKey:
    """Complete identity of inputs that may affect backend compilation."""

    registry_identity: ObjectIdentity
    backend: str
    collective: str
    requested_strategy: str
    executed_strategy: str
    fallback: tuple[str, ...]
    output_layout: str
    async_op: bool
    stage_signature: tuple[tuple[object, ...], ...]
    shape_class: tuple[int, ...]
    dtype: str
    layout: str
    rank: int
    world_size: int
    device: str
    local_rank: int | None
    local_world_size: int | None
    node_id: int | None
    node_count: int | None
    process_group_identity: ObjectIdentity
    process_group_identities: tuple[tuple[str, ObjectIdentity], ...]
    bit: int | None
    group_size: int | None
    compression: CompressionConfig | None
    topology_signature: str
    device_architecture: str
    strategy_policy_id: str | None
    workspace_budget_bytes: int | None
    allow_dynamic_shape: bool
    workspace_policy: WorkspacePolicy


@dataclass(frozen=True)
class CompiledCommunicationPlan:
    """Bind a reusable executor to immutable compile-time metadata."""

    executor: CompiledExecutor
    execution_info: ExecutionInfo
    cache_key: CompileCacheKey

    def run(self, tensor: object, *, out: object | None = None) -> CollectiveWork[object]:
        """Execute the precompiled data path without control-plane lookup."""

        if out is None:
            return self.executor.run(tensor)
        if self.cache_key.collective != "reduce_scatter" or self.cache_key.output_layout != "shard":
            raise TypeError("caller-owned shard output is supported only by reduce_scatter shard plans")
        return self.executor.run(tensor, out=out)
