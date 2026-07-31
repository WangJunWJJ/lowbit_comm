from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_all_gather, compressed_all_reduce
from tests.benchmarks.result_schema import validate_result


def parse_args() -> argparse.Namespace:
    """Parse distributed collective performance comparison arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compact", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def setup() -> tuple[int, int, torch.device]:
    """Initialize a local NCCL process group."""

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def dtype_from_name(name: str) -> torch.dtype:
    """Return a torch dtype for a CCDL dtype name."""

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def benchmark(fn, *, warmup: int, repeat: int, device: torch.device) -> tuple[float, int]:
    """Measure worst-rank latency and peak allocated CUDA memory."""

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
    """Return relative L2 error against a reference tensor."""

    return float((reference.float() - candidate.float()).norm() / reference.float().norm())


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


def git_commit() -> str:
    override = os.environ.get("CCDL_BENCHMARK_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "unknown"


def result_record(
    *,
    strategy: str,
    latency_ms: float,
    peak_memory_bytes: int,
    logical_bytes: int,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    args: argparse.Namespace,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    result: dict[str, object] = {
        "commit": git_commit(),
        "hostname": platform.node(),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": str(torch.version.cuda),
        "torch_version": torch.__version__,
        "world_size": world_size,
        "dtype": args.dtype,
        "numel": args.numel,
        "strategy": strategy,
        "latency_ms": latency_ms,
        "effective_gbps": logical_bytes / latency_ms / 1_000_000,
        "peak_memory_bytes": peak_memory_bytes,
        **error_metrics(reference, candidate),
    }
    validate_result(result)
    return result


def run() -> None:
    """Run baseline and CCDL compressed collective benchmarks."""

    args = parse_args()
    rank, world_size, device = setup()
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    source = torch.randn(args.numel, device=device, dtype=dtype)
    baseline_reference = source.clone()
    dist.all_reduce(baseline_reference, op=dist.ReduceOp.SUM)
    baseline_reference /= world_size
    gather_reference = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(gather_reference, source)

    def torch_all_reduce_once() -> None:
        tensor = source.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size

    def torch_all_gather_once() -> None:
        output = [torch.empty_like(source) for _ in range(world_size)]
        dist.all_gather(output, source)

    config = CompressionConfig(bit=args.bit, group_size=args.group_size, compact=args.compact)

    def ccdl_all_reduce_once() -> None:
        compressed_all_reduce(source.clone(), config=config, op="mean", strategy="all_gather", dtype=args.dtype)

    def ccdl_all_gather_once() -> None:
        compressed_all_gather(source.clone(), config=config, dtype=args.dtype)

    torch_ms, torch_peak = benchmark(torch_all_reduce_once, warmup=args.warmup, repeat=args.repeat, device=device)
    ccdl_ms, ccdl_peak = benchmark(ccdl_all_reduce_once, warmup=args.warmup, repeat=args.repeat, device=device)
    torch_gather_ms, torch_gather_peak = benchmark(
        torch_all_gather_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    ccdl_gather_ms, ccdl_gather_peak = benchmark(
        ccdl_all_gather_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    torch_result = source.clone()
    dist.all_reduce(torch_result, op=dist.ReduceOp.SUM)
    torch_result /= world_size
    torch_gather_result = [torch.empty_like(source) for _ in range(world_size)]
    dist.all_gather(torch_gather_result, source)
    ccdl_result = compressed_all_reduce(source.clone(), config=config, op="mean", strategy="all_gather", dtype=args.dtype)
    ccdl_gather_result = compressed_all_gather(source.clone(), config=config, dtype=args.dtype)
    torch.cuda.synchronize(device)
    error = relative_l2(baseline_reference, ccdl_result)
    gather_error = relative_l2(torch.cat(gather_reference), torch.cat(ccdl_gather_result))
    element_bytes = source.element_size()
    results = [
        result_record(
            strategy="torch_all_reduce",
            latency_ms=torch_ms,
            peak_memory_bytes=torch_peak,
            logical_bytes=args.numel * element_bytes,
            reference=baseline_reference,
            candidate=torch_result,
            args=args,
            world_size=world_size,
            device=device,
        ),
        result_record(
            strategy="ccdl_all_gather_reduce",
            latency_ms=ccdl_ms,
            peak_memory_bytes=ccdl_peak,
            logical_bytes=args.numel * element_bytes,
            reference=baseline_reference,
            candidate=ccdl_result,
            args=args,
            world_size=world_size,
            device=device,
        ),
        result_record(
            strategy="torch_all_gather",
            latency_ms=torch_gather_ms,
            peak_memory_bytes=torch_gather_peak,
            logical_bytes=args.numel * element_bytes * world_size,
            reference=torch.cat(gather_reference),
            candidate=torch.cat(torch_gather_result),
            args=args,
            world_size=world_size,
            device=device,
        ),
        result_record(
            strategy="ccdl_all_gather",
            latency_ms=ccdl_gather_ms,
            peak_memory_bytes=ccdl_gather_peak,
            logical_bytes=args.numel * element_bytes * world_size,
            reference=torch.cat(gather_reference),
            candidate=torch.cat(ccdl_gather_result),
            args=args,
            world_size=world_size,
            device=device,
        ),
    ]

    summary = {
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "world_size": world_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "compact": args.compact,
        "torch_all_reduce_ms": torch_ms,
        "ccdl_all_gather_reduce_ms": ccdl_ms,
        "latency_ratio_ccdl_over_torch": ccdl_ms / torch_ms,
        "relative_l2": error,
        "torch_all_gather_ms": torch_gather_ms,
        "ccdl_all_gather_ms": ccdl_gather_ms,
        "latency_ratio_ccdl_gather_over_torch": ccdl_gather_ms / torch_gather_ms,
        "all_gather_relative_l2": gather_error,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
