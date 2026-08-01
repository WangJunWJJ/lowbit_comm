from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.shard import ReducedShard
from ccdl_comm.executor import CompiledCommunicationPlan


def _compile_cuda_shortcut(tensor: Any, **kwargs: Any) -> CompiledCommunicationPlan:
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut

    return compile_cuda_shortcut(tensor, **kwargs)


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
    compiled_plan: CompiledCommunicationPlan | None = None,
) -> Any:
    """Return only this rank's reduced shard for sharded training consumers."""

    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(
            f"reduce_scatter_shard:{op}",
            reason="only op='sum' and op='mean' are implemented",
        )
    if compiled_plan is not None:
        work = compiled_plan.run(tensor)
        return work if async_op else work.wait()
    if reduce_scatter_shard is None and _is_cuda_tensor(tensor):
        compiled = _compile_cuda_shortcut(
            tensor,
            collective="reduce_scatter",
            strategy="compressed",
            output_layout="shard",
            config=config,
            async_op=async_op,
            dtype=dtype,
            extension_status=extension_status,
        )
        work = compiled.run(tensor)
        return work if async_op else work.wait()
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


def _is_cuda_tensor(tensor: Any) -> bool:
    return str(getattr(tensor, "device", "")).lower().startswith("cuda")
