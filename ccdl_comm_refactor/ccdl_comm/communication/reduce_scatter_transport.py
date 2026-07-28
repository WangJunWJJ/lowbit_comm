from __future__ import annotations

from collections.abc import Callable
from importlib import import_module as _import_module
from typing import Any

from ccdl_comm.collectives.reduce_scatter import ReducedShard
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_reduce_tensors, quantize_tensor


def make_torch_compressed_reduce_scatter_all_gather(
    *,
    import_module: Callable[[str], Any] = _import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize_reduce: Callable[..., Any] = dequantize_reduce_tensors,
) -> Callable[..., Any]:
    """Create a torch.distributed compressed reduce-scatter/full-gather transport.

    This prototype performs the performance-critical exchange as compressed
    all-to-all of per-destination bucket chunks, then restores full DDP bucket
    semantics by all-gathering the reduced full-precision shards.
    """

    shard_transport = make_torch_compressed_reduce_scatter_shard(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> Any:
        reduced = shard_transport(
            tensor,
            config=config,
            op=op,
            async_op=async_op,
            dtype=dtype,
            extension_status=extension_status,
        )
        dist = _distributed(import_module)
        torch = import_module("torch")
        restored_shards = [reduced.shard.new_empty((reduced.shard_numel,)) for _ in range(reduced.world_size)]
        dist.all_gather(restored_shards, reduced.shard)
        return torch.cat(restored_shards, dim=0).reshape(reduced.original_shape)

    return transport


def make_torch_compressed_reduce_scatter_shard(
    *,
    import_module: Callable[[str], Any] = _import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize_reduce: Callable[..., Any] = dequantize_reduce_tensors,
) -> Callable[..., ReducedShard]:
    """Create a torch.distributed transport that returns only the local shard."""

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> ReducedShard:
        if async_op:
            raise UnsupportedCollective("reduce_scatter:async", reason="reduce-scatter prototype is synchronous")
        if op not in {"sum", "mean"}:
            raise UnsupportedCollective(f"reduce_scatter:{op}", reason="only op='sum' and op='mean' are implemented")

        dist = _distributed(import_module)
        world_size = int(dist.get_world_size())
        flat = tensor.reshape((-1,))
        numel = int(flat.numel())
        if numel % world_size != 0:
            raise UnsupportedCollective(
                "reduce_scatter:shape",
                reason="flattened bucket numel must be divisible by world size",
            )
        rank = int(dist.get_rank())
        shard_numel = numel // world_size
        chunks = tuple(flat.chunk(world_size))
        compressed_chunks = [
            quantize(chunk, config, extension_status=extension_status)
            for chunk in chunks
        ]
        _require_equal_payload_shapes(compressed_chunks)

        received = [
            compressed_chunks[0].new_empty(tuple(compressed_chunks[0].shape))
            for _ in range(world_size)
        ]
        dist.all_to_all(received, compressed_chunks)
        reduced_shard = dequantize_reduce(
            received,
            (shard_numel,),
            config,
            dtype=dtype,
            extension_status=extension_status,
            reduce=op,
        )
        return ReducedShard(
            shard=reduced_shard,
            shard_index=rank,
            shard_numel=shard_numel,
            original_shape=tuple(tensor.shape),
            original_numel=numel,
            world_size=world_size,
            reduce=op,
        )

    return transport


def _distributed(import_module: Callable[[str], Any]) -> Any:
    try:
        dist = import_module("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError("torch.distributed is not available") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError("torch.distributed is not initialized")
    return dist


def _require_equal_payload_shapes(payloads: list[Any]) -> None:
    if not payloads:
        raise UnsupportedCollective("reduce_scatter:payload", reason="no compressed chunks were produced")
    expected = tuple(getattr(payloads[0], "shape", ()))
    for payload in payloads[1:]:
        if tuple(getattr(payload, "shape", ())) != expected:
            raise UnsupportedCollective(
                "reduce_scatter:payload",
                reason="compressed all-to-all prototype requires equal-size compressed chunks",
            )
