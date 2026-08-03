"""Immutable schedules and async submission for tree collective transports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ._executor_support import CompletionManager, ExecutorSupport, WorkspaceSession
from .compressed_reduce_scatter import ChunkPlan, ChunkRange


@dataclass(frozen=True, slots=True)
class TreeEdge:
    """A child-to-parent tree edge, used in reverse for broadcast."""

    child_rank: int
    parent_rank: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.child_rank, "child_rank")
        _require_nonnegative_integer(self.parent_rank, "parent_rank")
        if self.child_rank == self.parent_rank:
            raise ValueError("child_rank and parent_rank must differ")


@dataclass(frozen=True, slots=True)
class TreeSchedule:
    """Rank-local tree metadata with deterministic reduction and broadcast order."""

    chunk_plan: ChunkPlan
    rank: int
    root: int
    parent: int | None
    children: tuple[int, ...]
    reduce_edges: tuple[TreeEdge, ...]
    broadcast_edges: tuple[TreeEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_plan, ChunkPlan):
            raise TypeError("chunk_plan must be a ChunkPlan")
        world_size = self.chunk_plan.world_size
        _require_rank(self.rank, world_size, "rank")
        _require_rank(self.root, world_size, "root")
        expected_edges = _tree_edges(world_size, self.root)
        if self.reduce_edges != expected_edges:
            raise ValueError("reduce_edges must match the deterministic tree topology")
        if self.broadcast_edges != tuple(reversed(expected_edges)):
            raise ValueError("broadcast_edges must reverse the reduce edge order")
        expected_parent = next(
            (edge.parent_rank for edge in expected_edges if edge.child_rank == self.rank),
            None,
        )
        if self.parent != expected_parent:
            raise ValueError("parent must match the deterministic tree topology")
        expected_children = tuple(
            edge.child_rank for edge in expected_edges if edge.parent_rank == self.rank
        )
        if self.children != expected_children:
            raise ValueError("children must match the deterministic tree topology")

    @property
    def local_chunk(self) -> ChunkRange:
        """The preplanned shard range local to this rank."""

        return self.chunk_plan.chunk_for_rank(self.rank)


class TreeRuntime(Protocol):
    """Pre-bound coarse operations required by :class:`TreeExecutor`."""

    def create_submission_context(self, tensor: Any) -> Any: ...

    def wait_for_producer(self, tensor: Any, *, context: Any) -> None: ...

    def quant_pack(
        self,
        tensor: Any,
        edge: TreeEdge,
        workspace: WorkspaceSession,
        *,
        context: Any,
    ) -> Any: ...

    def send(
        self,
        payload: Any,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: WorkspaceSession,
        context: Any,
    ) -> Any: ...

    def receive(
        self,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: WorkspaceSession,
        context: Any,
    ) -> tuple[Any, Any]: ...

    def fused_reduce(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: WorkspaceSession,
        *,
        context: Any,
        dependency: Any,
    ) -> Any: ...

    def apply_broadcast(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: WorkspaceSession,
        *,
        context: Any,
        dependency: Any,
    ) -> Any: ...

    def record_completion(
        self, *, context: Any, dependencies: tuple[Any, ...]
    ) -> Any: ...


class TreeExecutor:
    """Submit rank-local tree reduce/broadcast operations without blocking."""

    __slots__ = (
        "schedule",
        "_runtime",
        "_support",
    )

    def __init__(
        self,
        *,
        schedule: TreeSchedule,
        runtime: TreeRuntime,
        workspace_session_factory: Callable[[Any], WorkspaceSession],
        completion_manager: CompletionManager,
    ) -> None:
        if not isinstance(schedule, TreeSchedule):
            raise TypeError("schedule must be a TreeSchedule")
        if not callable(workspace_session_factory):
            raise TypeError("workspace_session_factory must be callable")
        self.schedule = schedule
        self._runtime = runtime
        self._support = ExecutorSupport(workspace_session_factory, completion_manager)

    @property
    def pending_submission_count(self) -> int:
        return self._support.pending_count

    def reap_pending(self) -> None:
        self._support.reap()

    def run(self, tensor: Any) -> Any:
        """Enqueue reduce edges, reverse broadcast edges, and completion."""

        self._support.reap()
        owner = None
        try:
            owner = self._support.begin(tensor, self._runtime)
            workspace = owner.workspace
            context = owner.context
            self._runtime.wait_for_producer(tensor, context=context)
            for edge in self.schedule.reduce_edges:
                if edge.parent_rank == self.schedule.rank:
                    received, handle = self._runtime.receive(
                        peer=edge.child_rank,
                        edge=edge,
                        workspace=workspace,
                        context=context,
                    )
                    owner.retain(received)
                    owner.depend_on(handle)
                    reduction = self._runtime.fused_reduce(
                        tensor,
                        received,
                        edge,
                        workspace,
                        context=context,
                        dependency=handle,
                    )
                    owner.depend_on(reduction)
                elif edge.child_rank == self.schedule.rank:
                    payload = self._runtime.quant_pack(
                        tensor, edge, workspace, context=context
                    )
                    handle = self._runtime.send(
                        payload,
                        peer=edge.parent_rank,
                        edge=edge,
                        workspace=workspace,
                        context=context,
                    )
                    owner.retain(payload)
                    owner.depend_on(handle)

            for edge in self.schedule.broadcast_edges:
                if edge.child_rank == self.schedule.rank:
                    received, handle = self._runtime.receive(
                        peer=edge.parent_rank,
                        edge=edge,
                        workspace=workspace,
                        context=context,
                    )
                    owner.retain(received)
                    owner.depend_on(handle)
                    broadcast = self._runtime.apply_broadcast(
                        tensor,
                        received,
                        edge,
                        workspace,
                        context=context,
                        dependency=handle,
                    )
                    owner.depend_on(broadcast)
                elif edge.parent_rank == self.schedule.rank:
                    payload = self._runtime.quant_pack(
                        tensor, edge, workspace, context=context
                    )
                    handle = self._runtime.send(
                        payload,
                        peer=edge.child_rank,
                        edge=edge,
                        workspace=workspace,
                        context=context,
                    )
                    owner.retain(payload)
                    owner.depend_on(handle)

            self._support.record(owner, self._runtime)
            return self._support.finish(owner)
        except BaseException:
            if owner is not None:
                try:
                    self._support.fail(owner, self._runtime)
                except BaseException:
                    pass
            raise


def compile_tree_schedule(*, chunk_plan: ChunkPlan, rank: int, root: int = 0) -> TreeSchedule:
    """Compile a root-relative binary tree for any positive world size."""

    if not isinstance(chunk_plan, ChunkPlan):
        raise TypeError("chunk_plan must be a ChunkPlan")
    world_size = chunk_plan.world_size
    _require_rank(rank, world_size, "rank")
    _require_rank(root, world_size, "root")
    reduce_edges = _tree_edges(world_size, root)
    parent = next((edge.parent_rank for edge in reduce_edges if edge.child_rank == rank), None)
    children = tuple(edge.child_rank for edge in reduce_edges if edge.parent_rank == rank)
    return TreeSchedule(
        chunk_plan=chunk_plan,
        rank=rank,
        root=root,
        parent=parent,
        children=children,
        reduce_edges=reduce_edges,
        broadcast_edges=tuple(reversed(reduce_edges)),
    )


def _tree_edges(world_size: int, root: int) -> tuple[TreeEdge, ...]:
    logical_to_rank = tuple((root + logical_rank) % world_size for logical_rank in range(world_size))
    return tuple(
        TreeEdge(
            child_rank=logical_to_rank[logical_rank],
            parent_rank=logical_to_rank[(logical_rank - 1) // 2],
        )
        for logical_rank in range(world_size - 1, 0, -1)
    )


def _require_rank(value: object, world_size: int, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value >= world_size:
        raise ValueError(f"{name} must be in [0, {world_size})")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
