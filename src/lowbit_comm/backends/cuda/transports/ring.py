"""Immutable general ring schedules with no runtime policy decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RingStep:
    send_peer: int
    recv_peer: int
    send_chunk: int
    recv_chunk: int


@dataclass(frozen=True, slots=True)
class RingSchedule:
    rank: int
    world_size: int
    reduce_scatter: tuple[RingStep, ...]
    all_gather: tuple[RingStep, ...]


def compile_ring_schedule(*, world_size: int, rank: int) -> RingSchedule:
    _validate(world_size, rank)
    successor = (rank + 1) % world_size
    predecessor = (rank - 1) % world_size
    reduce_scatter = tuple(
        RingStep(
            send_peer=successor,
            recv_peer=predecessor,
            send_chunk=(rank - step) % world_size,
            recv_chunk=(rank - step - 1) % world_size,
        )
        for step in range(world_size - 1)
    )
    all_gather = tuple(
        RingStep(
            send_peer=successor,
            recv_peer=predecessor,
            send_chunk=(rank - step + 1) % world_size,
            recv_chunk=(rank - step) % world_size,
        )
        for step in range(world_size - 1)
    )
    return RingSchedule(rank, world_size, reduce_scatter, all_gather)


def _validate(world_size: int, rank: int) -> None:
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within world_size")
