from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective


def compressed_hierarchical_all_reduce(
    tensor: Any,
    *,
    config: CompressionConfig,
    op: str = "mean",
    async_op: bool = False,
    dtype: str = "auto",
    hierarchical_all_reduce: Callable[..., Any] | None = None,
    all_gather_fallback: Callable[..., Any] | None = None,
    extension_status: Any | None = None,
) -> Any:
    """Capability-gated hierarchical compressed all-reduce prototype.

    This entry point establishes the transport contract for future intra-node /
    inter-node hierarchical compression. It is safe by default: without an
    injected hierarchical transport it uses the caller-provided all-gather
    fallback or raises a clear unsupported-transport error.
    """

    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(f"hierarchical:{op}", reason="only op='sum' and op='mean' are implemented")

    if hierarchical_all_reduce is not None:
        return hierarchical_all_reduce(
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
        "hierarchical:transport",
        reason="hierarchical compressed transport is unavailable and no all-gather fallback was provided",
    )
