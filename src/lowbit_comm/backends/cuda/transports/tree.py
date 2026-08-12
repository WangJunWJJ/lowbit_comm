"""Immutable root-relative binary-tree schedules for arbitrary world sizes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TreeSchedule:
    rank: int
    world_size: int
    root: int
    parent: int | None
    children: tuple[int, ...]


def compile_tree_schedule(
    *, world_size: int, rank: int, root: int = 0
) -> TreeSchedule:
    _rank(world_size, rank, "rank")
    _rank(world_size, root, "root")
    logical = (rank - root) % world_size
    parent = None if logical == 0 else ((logical - 1) // 2 + root) % world_size
    children = tuple(
        (child + root) % world_size
        for child in (2 * logical + 1, 2 * logical + 2)
        if child < world_size
    )
    return TreeSchedule(rank, world_size, root, parent, children)


def _rank(world_size: int, rank: int, name: str) -> None:
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"{name} must be an integer")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"{name} must be within world_size")
