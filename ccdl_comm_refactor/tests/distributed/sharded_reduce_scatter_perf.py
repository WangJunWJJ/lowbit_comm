from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_reduce_scatter_shard
from ccdl_comm.communication import make_native_topology_reduce_scatter_shard
from ccdl_comm.communication.reduce_scatter_transport import make_torch_compressed_reduce_scatter_shard
from tests.benchmarks.result_schema import resolve_benchmark_identity, validate_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=16_777_216)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--transport", choices=("all_to_all", "topology"), default="all_to_all")
    parser.add_argument("--topology-method", choices=("auto", "p2p", "ring"), default="auto")
    return parser.parse_args()


def setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def benchmark(fn, *, warmup: int, repeat: int, device: torch.device) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000 / repeat
    peak_memory = torch.cuda.max_memory_allocated(device)
    measurements = torch.tensor([latency_ms, float(peak_memory)], dtype=torch.float64, device=device)
    dist.all_reduce(measurements, op=dist.ReduceOp.MAX)
    return float(measurements[0].item()), int(measurements[1].item())


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    return float((reference_f32 - candidate_f32).norm() / reference_f32.norm().clamp_min(1e-12))


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = reference_f32 - candidate_f32
    return {
        "relative_l2": float(difference.norm() / reference_f32.norm().clamp_min(1e-12)),
        "max_abs_error": float(difference.abs().max()),
        "rmse": float(difference.square().mean().sqrt()),
        "non_finite": int((~torch.isfinite(candidate_f32)).sum().item()),
    }


def result_record(
    *,
    strategy: str,
    latency_ms: float,
    peak_memory_bytes: int,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    args: argparse.Namespace,
    world_size: int,
    device: torch.device,
    identity: dict[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {
        **identity,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": str(torch.version.cuda),
        "torch_version": torch.__version__,
        "world_size": world_size,
        "dtype": args.dtype,
        "numel": args.numel,
        "strategy": strategy,
        "latency_ms": latency_ms,
        "effective_gbps": args.numel * candidate.element_size() / latency_ms / 1_000_000,
        "peak_memory_bytes": peak_memory_bytes,
        **error_metrics(reference, candidate),
    }
    validate_result(result)
    return result


def run() -> None:
    args = parse_args()
    rank, world_size, device = setup()
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    identity = resolve_benchmark_identity()
    source = torch.randn(args.numel, device=device, dtype=dtype)
    shard_numel = (args.numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    if padded_numel != args.numel:
        source_for_reference = torch.cat((source, source.new_zeros((padded_numel - args.numel,))), dim=0)
    else:
        source_for_reference = source

    full_reference = source_for_reference.clone()
    dist.all_reduce(full_reference, op=dist.ReduceOp.SUM)
    full_reference /= world_size
    reference_shard = full_reference.narrow(0, rank * shard_numel, shard_numel).contiguous()
    torch.cuda.synchronize(device)

    def torch_reduce_scatter_once() -> torch.Tensor:
        full = source_for_reference.clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM)
        full /= world_size
        return full.narrow(0, rank * shard_numel, shard_numel).contiguous()

    shard_transport = (
        make_native_topology_reduce_scatter_shard(
            method=None if args.topology_method == "auto" else args.topology_method
        )
        if args.transport == "topology"
        else make_torch_compressed_reduce_scatter_shard()
    )
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)

    def ccdl_shard_once() -> torch.Tensor:
        shard = compressed_reduce_scatter_shard(
            source,
            config=config,
            op="mean",
            dtype=args.dtype,
            reduce_scatter_shard=shard_transport,
        )
        return shard.shard

    torch_ms, torch_peak = benchmark(
        torch_reduce_scatter_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    ccdl_ms, ccdl_peak = benchmark(ccdl_shard_once, warmup=args.warmup, repeat=args.repeat, device=device)
    torch_result = torch_reduce_scatter_once()
    ccdl_result = ccdl_shard_once()
    torch.cuda.synchronize(device)
    error = relative_l2(reference_shard, ccdl_result)
    results = [
        result_record(
            strategy="torch_all_reduce_shard",
            latency_ms=torch_ms,
            peak_memory_bytes=torch_peak,
            reference=reference_shard,
            candidate=torch_result,
            args=args,
            world_size=world_size,
            device=device,
            identity=identity,
        ),
        result_record(
            strategy=f"ccdl_compressed_reduce_scatter_{args.transport}",
            latency_ms=ccdl_ms,
            peak_memory_bytes=ccdl_peak,
            reference=reference_shard,
            candidate=ccdl_result,
            args=args,
            world_size=world_size,
            device=device,
            identity=identity,
        ),
    ]

    summary = {
        "world_size": world_size,
        "numel": args.numel,
        "padded_numel": padded_numel,
        "shard_numel": shard_numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "transport": args.transport,
        "topology_method": args.topology_method if args.transport == "topology" else None,
        "torch_reduce_scatter_ms": torch_ms,
        "ccdl_shard_ms": ccdl_ms,
        "latency_ratio_ccdl_over_torch": ccdl_ms / torch_ms,
        "relative_l2": error,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
        "benchmark_identity": identity,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
