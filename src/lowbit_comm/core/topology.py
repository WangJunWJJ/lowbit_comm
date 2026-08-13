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


@dataclass(frozen=True, slots=True)
class GroupedReductionPlan:
    world_size: int
    max_fan_in: int
    levels: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "levels",
            tuple(tuple(tuple(group) for group in level) for level in self.levels),
        )
        if self.world_size <= 0 or self.max_fan_in <= 1:
            raise ValueError("grouped reduction dimensions must be positive")
        if not self.levels or len(self.levels[-1]) != 1:
            raise ValueError("grouped reduction must terminate at one root group")
        if any(
            not group or len(group) > self.max_fan_in
            for level in self.levels
            for group in level
        ):
            raise ValueError("grouped reduction group exceeds max_fan_in")
        first_ranks = tuple(rank for group in self.levels[0] for rank in group)
        if sorted(first_ranks) != list(range(self.world_size)):
            raise ValueError("first reduction level must cover every rank exactly once")
        for previous, current in zip(self.levels, self.levels[1:]):
            expected = sorted(group[0] for group in previous)
            actual = sorted(rank for group in current for rank in group)
            if actual != expected:
                raise ValueError(
                    "reduction levels must consume every prior representative once"
                )

    @property
    def root(self) -> int:
        return self.levels[-1][0][0]
