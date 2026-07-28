from __future__ import annotations

from collections.abc import Callable
from importlib import import_module as _import_module
from typing import Any

from ccdl_comm.collectives.reduce_scatter import ReducedShard
from ccdl_comm.communication.workspace import ShardCommunicationWorkspaceCache
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_reduce_tensors, quantize_tensor


def make_torch_compressed_reduce_scatter_all_gather(
    *,
    import_module: Callable[[str], Any] = _import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize_reduce: Callable[..., Any] = dequantize_reduce_tensors,
    allocate_reduced_shard_workspace: Callable[[Any, tuple[int, ...], CompressionConfig], Any] | None = None,
    allocate_quantized_chunk_workspace: Callable[[Any, CompressionConfig], Any] | None = None,
    allocate_received_payload_workspace: Callable[[Any, int, int, CompressionConfig], Any] | None = None,
    workspace_cache: ShardCommunicationWorkspaceCache | None = None,
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
        allocate_reduced_shard_workspace=allocate_reduced_shard_workspace,
        allocate_quantized_chunk_workspace=allocate_quantized_chunk_workspace,
        allocate_received_payload_workspace=allocate_received_payload_workspace,
        workspace_cache=workspace_cache,
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
        restored = torch.cat(restored_shards, dim=0)
        return _trim_to_numel(restored, reduced.original_numel).reshape(reduced.original_shape)

    return transport


def make_torch_compressed_reduce_scatter_shard(
    *,
    import_module: Callable[[str], Any] = _import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize_reduce: Callable[..., Any] = dequantize_reduce_tensors,
    allocate_reduced_shard_workspace: Callable[[Any, tuple[int, ...], CompressionConfig], Any] | None = None,
    allocate_quantized_chunk_workspace: Callable[[Any, CompressionConfig], Any] | None = None,
    allocate_received_payload_workspace: Callable[[Any, int, int, CompressionConfig], Any] | None = None,
    workspace_cache: ShardCommunicationWorkspaceCache | None = None,
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
        torch = import_module("torch")
        world_size = int(dist.get_world_size())
        flat = tensor.reshape((-1,))
        numel = int(flat.numel())
        padded_flat = _pad_flat_to_world_size(flat, world_size, torch)
        padded_numel = int(padded_flat.numel())
        rank = int(dist.get_rank())
        shard_numel = padded_numel // world_size
        bucket_key = _bucket_workspace_key(
            tensor,
            padded_numel=padded_numel,
            world_size=world_size,
            dtype=dtype,
        )
        chunks = tuple(padded_flat.chunk(world_size))
        compressed_chunks = [
            _quantize_chunk(
                chunk,
                index,
                config,
                quantize=quantize,
                extension_status=extension_status,
                allocator=allocate_quantized_chunk_workspace,
                workspace_cache=workspace_cache,
                bucket_key=bucket_key,
                dtype=dtype,
                world_size=world_size,
            )
            for index, chunk in enumerate(chunks)
        ]
        _require_equal_payload_shapes(compressed_chunks)

        received = _allocate_received_payloads(
            compressed_chunks[0],
            world_size,
            config,
            allocator=allocate_received_payload_workspace,
            workspace_cache=workspace_cache,
            bucket_key=bucket_key,
        )
        dist.all_to_all(received, compressed_chunks)
        workspace_shape = (shard_numel,)
        output_workspace = _allocate_reduced_workspace(
            tensor,
            workspace_shape,
            config,
            dtype=dtype,
            world_size=world_size,
            rank=rank,
            allocator=allocate_reduced_shard_workspace,
            workspace_cache=workspace_cache,
            bucket_key=bucket_key,
        )
        if output_workspace is None:
            reduced_shard = dequantize_reduce(
                received,
                workspace_shape,
                config,
                dtype=dtype,
                extension_status=extension_status,
                reduce=op,
            )
        else:
            reduced_shard = dequantize_reduce(
                received,
                workspace_shape,
                config,
                dtype=dtype,
                extension_status=extension_status,
                reduce=op,
                output=output_workspace,
            )
        return ReducedShard(
            shard=reduced_shard,
            shard_index=rank,
            shard_numel=shard_numel,
            original_shape=tuple(tensor.shape),
            original_numel=numel,
            world_size=world_size,
            reduce=op,
            padded_numel=padded_numel,
            dtype=dtype,
            transport="compressed_all_to_all",
            metadata={
                "compression_bit": config.bit,
                "group_size": config.group_size,
                "workspace_output": output_workspace is not None,
                "workspace_shape": workspace_shape,
                "quantized_workspace_output": allocate_quantized_chunk_workspace is not None or workspace_cache is not None,
                "received_workspace_output": allocate_received_payload_workspace is not None or workspace_cache is not None,
                "workspace_cache": workspace_cache is not None,
            },
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


def _quantize_chunk(
    chunk: Any,
    index: int,
    config: CompressionConfig,
    *,
    quantize: Callable[..., Any],
    extension_status: Any | None,
    allocator: Callable[[Any, CompressionConfig], Any] | None,
    workspace_cache: ShardCommunicationWorkspaceCache | None,
    bucket_key: Any,
    dtype: str,
    world_size: int,
) -> Any:
    if allocator is None and workspace_cache is None:
        return quantize(chunk, config, extension_status=extension_status)
    output = (
        allocator(chunk, config)
        if allocator is not None
        else workspace_cache.get_quantized_chunk(bucket_key, index, chunk, config, dtype=dtype, world_size=world_size)
    )
    return quantize(chunk, config, extension_status=extension_status, output=output)


def _allocate_received_payloads(
    template: Any,
    world_size: int,
    config: CompressionConfig,
    *,
    allocator: Callable[[Any, int, int, CompressionConfig], Any] | None,
    workspace_cache: ShardCommunicationWorkspaceCache | None,
    bucket_key: Any,
) -> list[Any]:
    if allocator is None and workspace_cache is None:
        return [template.new_empty(tuple(template.shape)) for _ in range(world_size)]
    if allocator is not None:
        return [allocator(template, index, world_size, config) for index in range(world_size)]
    return [
        workspace_cache.get_received_payload(bucket_key, template, index, world_size=world_size, config=config)
        for index in range(world_size)
    ]


def _allocate_reduced_workspace(
    tensor: Any,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
    world_size: int,
    rank: int,
    allocator: Callable[[Any, tuple[int, ...], CompressionConfig], Any] | None,
    workspace_cache: ShardCommunicationWorkspaceCache | None,
    bucket_key: Any,
) -> Any | None:
    if allocator is not None:
        return allocator(tensor, shape, config)
    if workspace_cache is None:
        return None
    return workspace_cache.get_reduced_shard(
        bucket_key,
        tensor,
        shape,
        config,
        dtype=dtype,
        world_size=world_size,
        rank=rank,
    )


def _pad_flat_to_world_size(flat: Any, world_size: int, torch: Any) -> Any:
    numel = int(flat.numel())
    remainder = numel % world_size
    if remainder == 0:
        return flat
    padding = world_size - remainder
    zeros = flat.new_zeros((padding,))
    return torch.cat((flat, zeros), dim=0)


def _trim_to_numel(tensor: Any, numel: int) -> Any:
    flattened = tensor.reshape((-1,))
    try:
        return flattened[:numel]
    except TypeError:
        return flattened


def _bucket_workspace_key(tensor: Any, *, padded_numel: int, world_size: int, dtype: str) -> tuple[Any, ...]:
    return (
        tuple(getattr(tensor, "shape", ())),
        str(getattr(tensor, "dtype", "")),
        str(getattr(tensor, "device", "")),
        padded_numel,
        world_size,
        dtype,
    )
