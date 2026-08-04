"""Immutable planning for compressed reduce-scatter data movement."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkRange:
    """A half-open flattened tensor range owned by one destination rank."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_integer(self.start, "chunk start")
        _require_integer(self.stop, "chunk stop")
        if self.start < 0:
            raise ValueError("chunk start must be >= 0")
        if self.stop < self.start:
            raise ValueError("chunk stop must be >= start")

    @property
    def numel(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Compile-time description of equal-sized destination shard chunks."""

    original_numel: int
    world_size: int
    padded_numel: int
    shard_numel: int
    chunks: tuple[ChunkRange, ...]
    owner_by_chunk: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_integer(self.original_numel, "original_numel")
        _require_integer(self.world_size, "world_size")
        _require_integer(self.padded_numel, "padded_numel")
        _require_integer(self.shard_numel, "shard_numel")
        if self.original_numel < 0:
            raise ValueError("original_numel must be >= 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.shard_numel < 0:
            raise ValueError("shard_numel must be >= 0")
        if self.padded_numel != self.shard_numel * self.world_size:
            raise ValueError("padded_numel must equal shard_numel * world_size")
        if self.padded_numel < self.original_numel:
            raise ValueError("padded_numel must be >= original_numel")
        expected_chunks = tuple(
            ChunkRange(rank * self.shard_numel, (rank + 1) * self.shard_numel)
            for rank in range(self.world_size)
        )
        if self.chunks != expected_chunks:
            raise ValueError("chunks must be contiguous equal-sized rank ranges")
        for owner in self.owner_by_chunk:
            _require_integer(owner, "owner_by_chunk entry")
        if self.owner_by_chunk != tuple(range(self.world_size)):
            raise ValueError("owner_by_chunk must map each chunk to its destination rank")

    def chunk_for_rank(self, rank: int) -> ChunkRange:
        _require_integer(rank, "rank")
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        return self.chunks[rank]


def compile_chunk_plan(*, original_numel: int, world_size: int) -> ChunkPlan:
    """Compile a deterministic, topology-neutral equal-shard plan."""

    _require_integer(original_numel, "original_numel")
    _require_integer(world_size, "world_size")
    if original_numel < 0:
        raise ValueError("original_numel must be >= 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    shard_numel = (original_numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    chunks = tuple(
        ChunkRange(rank * shard_numel, (rank + 1) * shard_numel)
        for rank in range(world_size)
    )
    return ChunkPlan(
        original_numel=original_numel,
        world_size=world_size,
        padded_numel=padded_numel,
        shard_numel=shard_numel,
        chunks=chunks,
        owner_by_chunk=tuple(range(world_size)),
    )


def _require_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
