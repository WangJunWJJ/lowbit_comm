"""Compare FP16 and compressed ReducedShard restoration with NCCL."""

from __future__ import annotations

import argparse
import json
import os
import time
from statistics import median

import torch
import torch.distributed as dist

from ccdl_comm.communication.reduce_scatter_transport import (
    make_torch_compressed_reduce_scatter_all_gather,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
from ccdl_comm.cuda.workspace import CudaShardWorkspaceProvider, create_torch_workspace_pool
from ccdl_comm.quantization.codec import (
    inplace_dequantize_gathered,
    inplace_dequantize_reduce_mean_requantize,
    quantize_tensor,
)
from ccdl_comm.quantization.sizing import estimate_quantized_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    return parser.parse_args()


def _rank_difference(tensor: torch.Tensor) -> float:
    minimum = tensor.clone()
    maximum = tensor.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float((maximum - minimum).abs().max().item())


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _benchmark(transport, tensor: torch.Tensor, config: CompressionConfig, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        transport(
            tensor,
            config=config,
            op="mean",
            async_op=False,
            dtype="fp16",
            extension_status=None,
        )
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        transport(
            tensor,
            config=config,
            op="mean",
            async_op=False,
            dtype="fp16",
            extension_status=None,
        )
    torch.cuda.synchronize()
    dist.barrier()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations
    maximum = torch.tensor(elapsed_ms, device=tensor.device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def _reusing_full_output_allocator():
    workspace = None

    def allocate(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        nonlocal workspace
        required_numel = compile_chunk_plan(
            original_numel=tensor.numel(),
            world_size=world_size,
        ).padded_numel
        if workspace is None or workspace.numel() != required_numel:
            workspace = tensor.new_empty((required_numel,))
        return workspace

    return allocate


def _reusing_compressed_restore_allocators():
    requantized = None
    gathered = None
    allocations = 0

    def allocate_requantized(tensor, payload_numel, payload_stride, config):
        nonlocal requantized, allocations
        del payload_numel, config
        if requantized is None or requantized.numel() != payload_stride:
            requantized = tensor.new_empty((payload_stride,), dtype=torch.uint8)
            allocations += 1
        return requantized

    def allocate_gathered(payload, world_size):
        nonlocal gathered, allocations
        required_numel = payload.numel() * world_size
        if gathered is None or gathered.numel() != required_numel:
            gathered = payload.new_empty((required_numel,))
            allocations += 1
        return gathered

    def allocation_count() -> int:
        return allocations

    return allocate_requantized, allocate_gathered, allocation_count


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(20260806 + rank)
    tensor = torch.randn(args.numel, device="cuda", dtype=torch.float16)
    reference = tensor.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    reference.div_(world_size)

    config = CompressionConfig(bit=8, group_size=64, error_feedback=False)
    restore_dtypes: list[torch.dtype] = []

    def recording_restore_quantize(value, policy, *, extension_status=None):
        payload = quantize_tensor(value, policy, extension_status=extension_status)
        restore_dtypes.append(payload.dtype)
        return payload

    fp16_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="fp16",
        allocate_full_output_workspace=_reusing_full_output_allocator(),
    )
    compressed_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="compressed",
        restore_quantize=recording_restore_quantize,
        allocate_full_output_workspace=_reusing_full_output_allocator(),
    )
    pooled_compressed_pool = create_torch_workspace_pool(
        max_cached_bytes=512 * 1024 * 1024,
        max_entries=64,
    )
    pooled_compressed_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="compressed",
        workspace_cache=CudaShardWorkspaceProvider(
            pooled_compressed_pool,
            backend="cuda",
            collective="all_reduce",
            strategy="pooled_compressed_restore",
            device=str(tensor.device),
            pool_reduced_output=True,
        ),
        allocate_full_output_workspace=_reusing_full_output_allocator(),
    )
    fused_requantize_calls = 0
    fused_gathered_dequantize_calls = 0

    def fused_requantize(*call_args, **call_kwargs):
        nonlocal fused_requantize_calls
        used = inplace_dequantize_reduce_mean_requantize(*call_args, **call_kwargs)
        fused_requantize_calls += int(used)
        return used

    def fused_gathered_dequantize(*call_args, **call_kwargs):
        nonlocal fused_gathered_dequantize_calls
        used = inplace_dequantize_gathered(*call_args, **call_kwargs)
        fused_gathered_dequantize_calls += int(used)
        return used

    workspace_pool = create_torch_workspace_pool(
        max_cached_bytes=512 * 1024 * 1024,
        max_entries=64,
    )
    workspace_provider = CudaShardWorkspaceProvider(
        workspace_pool,
        backend="cuda",
        collective="all_reduce",
        strategy="fused_compressed_restore",
        device=str(tensor.device),
        pool_reduced_output=False,
    )
    pooled_fused_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="compressed",
        fused_restore_requantize=fused_requantize,
        fused_restore_dequantize=fused_gathered_dequantize,
        workspace_cache=workspace_provider,
        allocate_full_output_workspace=_reusing_full_output_allocator(),
    )
    allocate_requantized, allocate_gathered, direct_allocation_count = (
        _reusing_compressed_restore_allocators()
    )
    fused_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="compressed",
        fused_restore_requantize=fused_requantize,
        fused_restore_dequantize=fused_gathered_dequantize,
        allocate_requantized_restore_workspace=allocate_requantized,
        allocate_compressed_restore_workspace=allocate_gathered,
        allocate_full_output_workspace=_reusing_full_output_allocator(),
    )
    fp16_result = fp16_transport(
        tensor,
        config=config,
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )
    compressed_result = compressed_transport(
        tensor,
        config=config,
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )
    fused_result = fused_transport(
        tensor,
        config=config,
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )
    torch.cuda.synchronize()

    relative_l2 = _relative_l2(compressed_result, reference)
    additional_relative_l2 = _relative_l2(compressed_result, fp16_result)
    rank_max_difference = _rank_difference(compressed_result)
    fused_relative_l2 = _relative_l2(fused_result, reference)
    fused_additional_relative_l2 = _relative_l2(fused_result, compressed_result)
    fused_rank_max_difference = _rank_difference(fused_result)
    if restore_dtypes != [torch.uint8]:
        raise AssertionError(f"restore transport did not use exactly one uint8 payload: {restore_dtypes}")
    if rank_max_difference != 0.0:
        raise AssertionError(f"rank outputs differ by {rank_max_difference}")
    if fused_rank_max_difference != 0.0:
        raise AssertionError(f"fused rank outputs differ by {fused_rank_max_difference}")
    if fused_requantize_calls != 1 or fused_gathered_dequantize_calls != 1:
        raise AssertionError("fused restore kernels were not selected exactly once during correctness validation")
    if relative_l2 > args.max_relative_l2:
        raise AssertionError(f"relative L2 {relative_l2} exceeds {args.max_relative_l2}")

    if args.trials <= 0:
        raise ValueError("trials must be positive")
    paths = {
        "fp16": fp16_transport,
        "compressed": compressed_transport,
        "pooled_compressed": pooled_compressed_transport,
        "pooled_fused": pooled_fused_transport,
        "fused": fused_transport,
    }
    path_names = list(paths)
    trial_ms: dict[str, list[float]] = {name: [] for name in path_names}
    for trial in range(args.trials):
        offset = trial % len(path_names)
        trial_order = path_names[offset:] + path_names[:offset]
        for name in trial_order:
            trial_ms[name].append(
                _benchmark(paths[name], tensor, config, args.warmup, args.iterations)
            )
    fp16_ms = median(trial_ms["fp16"])
    compressed_ms = median(trial_ms["compressed"])
    pooled_compressed_ms = median(trial_ms["pooled_compressed"])
    pooled_fused_ms = median(trial_ms["pooled_fused"])
    fused_ms = median(trial_ms["fused"])
    shard_numel = (args.numel + world_size - 1) // world_size
    packed_bytes = estimate_quantized_size(shard_numel, dtype="fp16", config=config).quantized_bytes
    compressed_bytes = ((packed_bytes + 15) // 16) * 16
    evidence = {
        "world_size": world_size,
        "numel": args.numel,
        "restore_payload_dtype": "uint8",
        "fp16_restore_bytes_per_rank": shard_numel * 2,
        "compressed_restore_bytes_per_rank": compressed_bytes,
        "restore_compression_ratio": (shard_numel * 2) / compressed_bytes,
        "relative_l2_vs_exact": relative_l2,
        "additional_relative_l2_vs_fp16_restore": additional_relative_l2,
        "rank_max_difference": rank_max_difference,
        "fp16_restore_pipeline_ms": fp16_ms,
        "compressed_restore_pipeline_ms": compressed_ms,
        "pooled_compressed_restore_pipeline_ms": pooled_compressed_ms,
        "fused_restore_pipeline_ms": fused_ms,
        "pooled_fused_restore_pipeline_ms": pooled_fused_ms,
        "pipeline_speedup": fp16_ms / compressed_ms,
        "fused_speedup_vs_fp16": fp16_ms / fused_ms,
        "fused_speedup_vs_compressed": compressed_ms / fused_ms,
        "fused_speedup_vs_pooled_compressed": pooled_compressed_ms / fused_ms,
        "direct_fused_speedup_vs_pooled_fused": pooled_fused_ms / fused_ms,
        "fused_relative_l2_vs_exact": fused_relative_l2,
        "fused_additional_relative_l2_vs_compressed": fused_additional_relative_l2,
        "fused_rank_max_difference": fused_rank_max_difference,
        "fused_requantize_calls": fused_requantize_calls,
        "fused_gathered_dequantize_calls": fused_gathered_dequantize_calls,
        "workspace_pool_hits": workspace_pool.stats.hits,
        "workspace_pool_misses": workspace_pool.stats.misses,
        "direct_restore_workspace_allocations": direct_allocation_count(),
        "trial_ms": trial_ms,
    }
    if rank == 0:
        print(json.dumps(evidence, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
