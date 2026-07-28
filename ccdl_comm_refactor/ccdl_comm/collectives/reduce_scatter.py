from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    reduce_scatter_shard: Callable[..., ReducedShard] | None = None,
    extension_status: Any | None = None,
) -> ReducedShard:
    """Return only this rank's reduced shard for sharded training consumers."""

    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(
            f"reduce_scatter_shard:{op}",
            reason="only op='sum' and op='mean' are implemented",
        )
    if async_op:
        raise UnsupportedCollective(
            "reduce_scatter_shard:async",
            reason="compressed reduce-scatter shard transport is synchronous",
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
