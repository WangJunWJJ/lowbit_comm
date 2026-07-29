from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective


@dataclass(frozen=True)
class ReducedShard:
    """Local reduced shard for ParaScale/FSDP-style sharded consumers."""

    shard: Any
    shard_index: int
    shard_numel: int
    original_shape: tuple[int, ...]
    original_numel: int
    world_size: int
    reduce: str
    padded_numel: int | None = None
    dtype: str = "auto"
    layout: str = "flat_contiguous"
    transport: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.shard_index < 0:
            raise ValueError("shard_index must be >= 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.shard_index >= self.world_size:
            raise ValueError("shard_index must be < world_size")
        if self.shard_numel < 0:
            raise ValueError("shard_numel must be >= 0")
        if self.original_numel < 0:
            raise ValueError("original_numel must be >= 0")
        active_padded_numel = self.padded_numel if self.padded_numel is not None else self.shard_numel * self.world_size
        if active_padded_numel < self.original_numel:
            raise ValueError("padded_numel must be >= original_numel")
        object.__setattr__(self, "padded_numel", active_padded_numel)

    @property
    def shard_offset(self) -> int:
        return self.shard_index * self.shard_numel

    @property
    def shard_end(self) -> int:
        return min(self.shard_offset + self.shard_numel, self.original_numel)

    @property
    def valid_numel(self) -> int:
        return max(0, self.shard_end - self.shard_offset)

    @property
    def padding_numel(self) -> int:
        return self.shard_numel - self.valid_numel

    @property
    def logical_range(self) -> tuple[int, int]:
        return self.shard_offset, self.shard_end

    @property
    def has_padding(self) -> bool:
        return self.padding_numel > 0 or int(self.padded_numel or 0) > self.original_numel

    @property
    def is_padding_only(self) -> bool:
        return self.valid_numel == 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "shard_numel": self.shard_numel,
            "shard_offset": self.shard_offset,
            "shard_end": self.shard_end,
            "valid_numel": self.valid_numel,
            "original_shape": self.original_shape,
            "original_numel": self.original_numel,
            "padded_numel": self.padded_numel,
            "world_size": self.world_size,
            "reduce": self.reduce,
            "dtype": self.dtype,
            "layout": self.layout,
            "transport": self.transport,
            "metadata": dict(self.metadata),
        }


def compressed_reduce_scatter(
    tensor: Any,
    *,
    config: CompressionConfig,
    op: str = "mean",
    async_op: bool = False,
    dtype: str = "auto",
    reduce_scatter: Callable[..., Any] | None = None,
    all_gather_fallback: Callable[..., Any] | None = None,
    extension_status: Any | None = None,
) -> Any:
    """Capability-gated compressed reduce-scatter entry point.

    The first implementation is intentionally fallback-first. It establishes a
    public contract for future compressed reduce-scatter transports without
    changing the validated all-gather DDP path.
    """

    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(f"reduce_scatter:{op}", reason="only op='sum' and op='mean' are implemented")

    if reduce_scatter is not None:
        return reduce_scatter(
            tensor,
            config=config,
            op=op,
            async_op=async_op,
            dtype=dtype,
            extension_status=extension_status,
        )

    if all_gather_fallback is not None:
        return all_gather_fallback(
            tensor,
            config=config,
            op=op,
            async_op=async_op,
            dtype=dtype,
            extension_status=extension_status,
        )

    raise UnsupportedCollective(
        "reduce_scatter:transport",
        reason="compressed reduce-scatter transport is unavailable and no all-gather fallback was provided",
    )


def compressed_reduce_scatter_shard(
    tensor: Any,
    *,
    config: CompressionConfig,
    op: str = "mean",
    async_op: bool = False,
    dtype: str = "auto",
    reduce_scatter_shard: Callable[..., Any] | None = None,
    extension_status: Any | None = None,
) -> Any:
    """Return only this rank's reduced shard for sharded training consumers."""

    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(
            f"reduce_scatter_shard:{op}",
            reason="only op='sum' and op='mean' are implemented",
        )
    if reduce_scatter_shard is None:
        raise UnsupportedCollective(
            "reduce_scatter_shard:transport",
            reason="compressed reduce-scatter shard transport is unavailable",
        )
    return reduce_scatter_shard(
        tensor,
        config=config,
        op=op,
        async_op=async_op,
        dtype=dtype,
        extension_status=extension_status,
    )
