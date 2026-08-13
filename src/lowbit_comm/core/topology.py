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


def parse_topology_signature(signature: str, *, world_size: int) -> Topology:
    """Parse a stable node-id signature into backend-neutral topology facts."""

    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not isinstance(signature, str) or not signature.startswith("node_ids="):
        raise ValueError("topology signature must use node_ids=<comma-separated ids>")
    encoded = signature.removeprefix("node_ids=")
    try:
        node_ids = tuple(int(value.strip()) for value in encoded.split(","))
    except ValueError as error:
        raise ValueError("topology node ids must be integers") from error
    if len(node_ids) != world_size:
        raise ValueError(
            f"topology rank count {len(node_ids)} does not match world_size {world_size}"
        )
    if any(node_id < 0 for node_id in node_ids):
        raise ValueError("topology node ids must be non-negative")
    ordered_ids = tuple(dict.fromkeys(node_ids))
    groups = tuple(
        tuple(rank for rank, active in enumerate(node_ids) if active == node_id)
        for node_id in ordered_ids
    )
    return Topology(
        node_ids=node_ids,
        node_groups=groups,
        leaders=tuple(group[0] for group in groups),
    )


def compile_grouped_reduction(
    topology: Topology,
    *,
    max_fan_in: int,
) -> GroupedReductionPlan:
    """Compile a bounded-fan-in reduction tree without runtime dependencies."""

    if not isinstance(topology, Topology):
        raise TypeError("topology must be a Topology")
    if isinstance(max_fan_in, bool) or not isinstance(max_fan_in, int):
        raise TypeError("max_fan_in must be an integer")
    if max_fan_in <= 1:
        raise ValueError("max_fan_in must be greater than one")
    first = tuple(
        chunk
        for node_group in topology.node_groups
        for chunk in _chunks(node_group, max_fan_in)
    )
    levels = [first]
    representatives = tuple(group[0] for group in first)
    while len(representatives) > 1:
        level = _chunks(representatives, max_fan_in)
        levels.append(level)
        representatives = tuple(group[0] for group in level)
    return GroupedReductionPlan(
        world_size=topology.world_size,
        max_fan_in=max_fan_in,
        levels=tuple(levels),
    )


def _chunks(values: tuple[int, ...], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))
