"""Compare FP16 and compressed ReducedShard restoration with NCCL."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from ccdl_comm.communication.reduce_scatter_transport import (
    make_torch_compressed_reduce_scatter_all_gather,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.codec import quantize_tensor
from ccdl_comm.quantization.sizing import estimate_quantized_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
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

    fp16_transport = make_torch_compressed_reduce_scatter_all_gather(restore_mode="fp16")
    compressed_transport = make_torch_compressed_reduce_scatter_all_gather(
        restore_mode="compressed",
        restore_quantize=recording_restore_quantize,
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
    torch.cuda.synchronize()

    relative_l2 = _relative_l2(compressed_result, reference)
    additional_relative_l2 = _relative_l2(compressed_result, fp16_result)
    rank_max_difference = _rank_difference(compressed_result)
    if restore_dtypes != [torch.uint8]:
        raise AssertionError(f"restore transport did not use exactly one uint8 payload: {restore_dtypes}")
    if rank_max_difference != 0.0:
        raise AssertionError(f"rank outputs differ by {rank_max_difference}")
    if relative_l2 > args.max_relative_l2:
        raise AssertionError(f"relative L2 {relative_l2} exceeds {args.max_relative_l2}")

    fp16_ms = _benchmark(fp16_transport, tensor, config, args.warmup, args.iterations)
    compressed_ms = _benchmark(compressed_transport, tensor, config, args.warmup, args.iterations)
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
        "pipeline_speedup": fp16_ms / compressed_ms,
    }
    if rank == 0:
        print(json.dumps(evidence, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
