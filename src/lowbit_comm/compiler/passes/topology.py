"""Pure topology parsing for compile-time communication lowering."""

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


def parse_topology_signature(signature: str, *, world_size: int) -> Topology:
    """Parse ``node_ids=<id per global rank>`` into immutable rank groups."""

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
