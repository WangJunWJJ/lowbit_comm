from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

try:
    import torch
    import torch.distributed as dist
except (ImportError, ModuleNotFoundError):
    torch = None
    dist = None

from ccdl_comm import CompressionConfig, compressed_all_reduce


def build_parser() -> argparse.ArgumentParser:
    """Build the asynchronous completion benchmark argument parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("all_gather_reduce", "topology"), default="all_gather_reduce")
    parser.add_argument(
        "--topology-method",
        choices=("overlap-gather", "overlap-p2p", "overlap-tree", "overlap-scale"),
        default="overlap-gather",
    )
    parser.add_argument("--numel", type=int, default=4_194_304)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--compute-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def make_summary(
    *,
    mode: str,
    world_size: int,
    sync_ms: float,
    async_wait_ms: float,
    async_overlap_ms: float,
    compute_ms: float,
    launch_us: float,
    relative_l2: float,
    max_abs_error: float,
) -> dict[str, Any]:
    """Build comparable latency, overlap, and accuracy metrics."""

    overlap_denominator = min(sync_ms, compute_ms)
    overlap_efficiency = 0.0
    if overlap_denominator > 0:
        overlap_efficiency = (sync_ms + compute_ms - async_overlap_ms) / overlap_denominator
    return {
        "mode": mode,
        "world_size": world_size,
        "sync_ms": sync_ms,
        "async_wait_ms": async_wait_ms,
        "async_overlap_ms": async_overlap_ms,
        "compute_ms": compute_ms,
        "async_launch_us": launch_us,
        "async_speedup_over_sync": sync_ms / async_wait_ms,
        "overlap_efficiency": overlap_efficiency,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
    }


def _setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def _dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    consume: Callable[[Any], None],
    device: torch.device,
) -> float:
    for _ in range(warmup):
        consume(operation())
    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(repeat):
        consume(operation())
    torch.cuda.synchronize(device)
    dist.barrier()
    return (time.perf_counter() - start) * 1000 / repeat


def _measure_launch(operation: Callable[[], Any], *, warmup: int, repeat: int, device: torch.device) -> float:
    for _ in range(warmup):
        operation().wait()
    torch.cuda.synchronize(device)
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        work = operation()
        elapsed += time.perf_counter() - start
        work.wait()
    torch.cuda.synchronize(device)
    return elapsed * 1_000_000 / repeat


def _errors(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    difference = reference.float() - candidate.float()
    relative_l2 = float(difference.norm() / reference.float().norm().clamp_min(1e-12))
    return relative_l2, float(difference.abs().max())


def run() -> None:
    """Run synchronous, asynchronous, and overlapped CCDL measurements."""

    if torch is None or dist is None:
        raise RuntimeError("async completion benchmark requires PyTorch with torch.distributed")
    args = build_parser().parse_args()
    rank, world_size, device = _setup()
    torch.manual_seed(args.seed + rank)
    source = torch.randn(args.numel, device=device, dtype=_dtype(args.dtype))
    reference = source.clone()
    dist.all_reduce(reference)
    reference /= world_size
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)
    strategy = "topology" if args.mode == "topology" else "all_gather"

    def synchronous() -> torch.Tensor:
        return compressed_all_reduce(
            source.clone(),
            config=config,
            op="mean",
            strategy=strategy,
            dtype=args.dtype,
            topology_method=args.topology_method,
        )

    def asynchronous() -> Any:
        return compressed_all_reduce(
            source.clone(),
            config=config,
            op="mean",
            strategy=strategy,
            async_op=True,
            dtype=args.dtype,
            topology_method=args.topology_method,
        )

    compute_buffer = torch.randn(min(args.numel, 1_048_576), device=device, dtype=torch.float32)

    def compute() -> None:
        for _ in range(args.compute_iters):
            compute_buffer.sin_()

    sync_ms = _measure(synchronous, warmup=args.warmup, repeat=args.repeat, consume=lambda result: None, device=device)
    async_wait_ms = _measure(
        asynchronous,
        warmup=args.warmup,
        repeat=args.repeat,
        consume=lambda work: work.wait(),
        device=device,
    )
    compute_ms = _measure(
        lambda: compute(),
        warmup=args.warmup,
        repeat=args.repeat,
        consume=lambda result: None,
        device=device,
    )

    def consume_with_compute(work: Any) -> None:
        compute()
        work.wait()

    async_overlap_ms = _measure(
        asynchronous,
        warmup=args.warmup,
        repeat=args.repeat,
        consume=consume_with_compute,
        device=device,
    )
    launch_us = _measure_launch(asynchronous, warmup=args.warmup, repeat=args.repeat, device=device)
    candidate = asynchronous().wait()
    torch.cuda.synchronize(device)
    relative_l2, max_abs_error = _errors(reference, candidate)
    summary = make_summary(
        mode=args.mode,
        world_size=world_size,
        sync_ms=sync_ms,
        async_wait_ms=async_wait_ms,
        async_overlap_ms=async_overlap_ms,
        compute_ms=compute_ms,
        launch_us=launch_us,
        relative_l2=relative_l2,
        max_abs_error=max_abs_error,
    )
    summary.update(
        {
            "topology_method": args.topology_method if args.mode == "topology" else None,
            "numel": args.numel,
            "dtype": args.dtype,
            "bit": args.bit,
            "group_size": args.group_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "compute_iters": args.compute_iters,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
