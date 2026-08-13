"""Backend-neutral immutable topology facts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topology:
    node_ids: tuple[int, ...]
    node_groups: tuple[tuple[int, ...], ...]
    leaders: tuple[int, ...]

    @property
    def world_size(self) -> int:
        return len(self.node_ids)

    def group_for_rank(self, rank: int) -> tuple[int, ...]:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if rank < 0 or rank >= self.world_size:
            raise ValueError("rank must be within world_size")
        node_id = self.node_ids[rank]
        return next(
            group for group in self.node_groups if self.node_ids[group[0]] == node_id
        )
