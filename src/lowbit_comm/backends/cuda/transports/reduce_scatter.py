"""Static layout planning for quantized reduce-scatter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShardPlan:
    original_numel: int
    rank: int
    world_size: int
    shard_numel: int
    padded_numel: int

    @property
    def logical_range(self) -> tuple[int, int]:
        start = self.rank * self.shard_numel
        return start, min(start + self.shard_numel, self.original_numel)


def compile_shard_plan(
    *,
    original_numel: int,
    rank: int,
    world_size: int,
    group_size: int,
) -> ShardPlan:
    if original_numel < 0:
        raise ValueError("original_numel must be non-negative")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank/world_size are invalid")
    logical_shard = (original_numel + world_size - 1) // world_size
    shard_numel = (
        ((logical_shard + group_size - 1) // group_size) * group_size
        if logical_shard
        else 0
    )
    return ShardPlan(
        original_numel=original_numel,
        rank=rank,
        world_size=world_size,
        shard_numel=shard_numel,
        padded_numel=shard_numel * world_size,
    )
