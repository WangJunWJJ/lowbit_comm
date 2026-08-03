"""Immutable schedules and async submission for tree collective transports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .compressed_reduce_scatter import ChunkPlan, ChunkRange


class _WorkspaceSession(Protocol):
    def release(self, *, completion: Any) -> None: ...


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

    def wait_for_producer(self, tensor: Any) -> None: ...

    def quant_pack(self, tensor: Any, edge: TreeEdge, workspace: _WorkspaceSession) -> Any: ...

    def send(
        self,
        payload: Any,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: _WorkspaceSession,
    ) -> Any: ...

    def receive(
        self,
        *,
        peer: int,
        edge: TreeEdge,
        workspace: _WorkspaceSession,
    ) -> tuple[Any, Any]: ...

    def fused_reduce(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: _WorkspaceSession,
    ) -> None: ...

    def apply_broadcast(
        self,
        tensor: Any,
        received: Any,
        edge: TreeEdge,
        workspace: _WorkspaceSession,
    ) -> None: ...

    def record_completion(self) -> Any: ...


class _CompletionManager(Protocol):
    def create_work(
        self,
        *,
        result: Any,
        completion: Any,
        resources: tuple[Any, ...],
    ) -> Any: ...


class TreeExecutor:
    """Submit rank-local tree reduce/broadcast operations without blocking."""

    __slots__ = (
        "schedule",
        "_runtime",
        "_workspace_session_factory",
        "_completion_manager",
    )

    def __init__(
        self,
        *,
        schedule: TreeSchedule,
        runtime: TreeRuntime,
        workspace_session_factory: Callable[[Any], _WorkspaceSession],
        completion_manager: _CompletionManager,
    ) -> None:
        if not isinstance(schedule, TreeSchedule):
            raise TypeError("schedule must be a TreeSchedule")
        if not callable(workspace_session_factory):
            raise TypeError("workspace_session_factory must be callable")
        self.schedule = schedule
        self._runtime = runtime
        self._workspace_session_factory = workspace_session_factory
        self._completion_manager = completion_manager

    def run(self, tensor: Any) -> Any:
        """Enqueue reduce edges, reverse broadcast edges, and completion."""

        workspace = self._workspace_session_factory(tensor)
        resources: list[Any] = [tensor, workspace]
        try:
            self._runtime.wait_for_producer(tensor)
            for edge in self.schedule.reduce_edges:
                if edge.parent_rank == self.schedule.rank:
                    received, handle = self._runtime.receive(
                        peer=edge.child_rank,
                        edge=edge,
                        workspace=workspace,
                    )
                    _retain_submission(resources, received, handle)
                    self._runtime.fused_reduce(tensor, received, edge, workspace)
                elif edge.child_rank == self.schedule.rank:
                    payload = self._runtime.quant_pack(tensor, edge, workspace)
                    handle = self._runtime.send(
                        payload,
                        peer=edge.parent_rank,
                        edge=edge,
                        workspace=workspace,
                    )
                    _retain_submission(resources, payload, handle)

            for edge in self.schedule.broadcast_edges:
                if edge.child_rank == self.schedule.rank:
                    received, handle = self._runtime.receive(
                        peer=edge.parent_rank,
                        edge=edge,
                        workspace=workspace,
                    )
                    _retain_submission(resources, received, handle)
                    self._runtime.apply_broadcast(tensor, received, edge, workspace)
                elif edge.parent_rank == self.schedule.rank:
                    payload = self._runtime.quant_pack(tensor, edge, workspace)
                    handle = self._runtime.send(
                        payload,
                        peer=edge.child_rank,
                        edge=edge,
                        workspace=workspace,
                    )
                    _retain_submission(resources, payload, handle)

            completion = self._runtime.record_completion()
        except BaseException:
            _abort_workspace(workspace, self._runtime)
            raise

        _release_workspace(workspace, completion)
        return self._completion_manager.create_work(
            result=tensor,
            completion=completion,
            resources=tuple(resources),
        )


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


def _retain_submission(resources: list[Any], payload: Any, handle: Any) -> None:
    resources.append(payload)
    if handle is not None:
        resources.append(handle)


def _release_workspace(workspace: _WorkspaceSession, completion: Any) -> None:
    release = getattr(workspace, "release", None)
    if not callable(release):
        raise TypeError("workspace session must provide release(completion=...)")
    release(completion=completion)


def _abort_workspace(workspace: _WorkspaceSession, runtime: TreeRuntime) -> None:
    abort = getattr(workspace, "abort", None)
    if callable(abort):
        abort()
        return
    completion = runtime.record_completion()
    _release_workspace(workspace, completion)


def _require_rank(value: object, world_size: int, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value >= world_size:
        raise ValueError(f"{name} must be in [0, {world_size})")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
