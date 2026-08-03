"""Immutable compiled schedules for tree-based collective transports."""

from __future__ import annotations

from dataclasses import dataclass

from .compressed_reduce_scatter import ChunkPlan, ChunkRange


@dataclass(frozen=True, slots=True)
class TreeEdge:
    """A child-to-parent tree edge, used in reverse for broadcast."""

    child_rank: int
    parent_rank: int


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value >= world_size:
        raise ValueError(f"{name} must be in [0, {world_size})")
