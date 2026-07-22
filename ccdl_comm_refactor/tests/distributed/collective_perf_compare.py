from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_all_reduce


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


def benchmark(fn, *, warmup: int, repeat: int, device: torch.device) -> float:
    """Measure average CUDA-synchronized latency in milliseconds."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeat


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return relative L2 error against a reference tensor."""

    return float((reference.float() - candidate.float()).norm() / reference.float().norm())


def run() -> None:
    """Run baseline and CCDL compressed all-reduce benchmarks."""

    args = parse_args()
    rank, world_size, device = setup()
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    source = torch.randn(args.numel, device=device, dtype=dtype)
    baseline_reference = source.clone()
    dist.all_reduce(baseline_reference, op=dist.ReduceOp.SUM)
    baseline_reference /= world_size

    def torch_all_reduce_once() -> None:
        tensor = source.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size

    config = CompressionConfig(bit=args.bit, group_size=args.group_size)

    def ccdl_all_reduce_once() -> None:
        compressed_all_reduce(source.clone(), config=config, op="mean", strategy="all_gather", dtype=args.dtype)

    torch_ms = benchmark(torch_all_reduce_once, warmup=args.warmup, repeat=args.repeat, device=device)
    ccdl_ms = benchmark(ccdl_all_reduce_once, warmup=args.warmup, repeat=args.repeat, device=device)
    ccdl_result = compressed_all_reduce(source.clone(), config=config, op="mean", strategy="all_gather", dtype=args.dtype)
    torch.cuda.synchronize(device)
    error = relative_l2(baseline_reference, ccdl_result)

    summary = {
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "world_size": world_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "torch_all_reduce_ms": torch_ms,
        "ccdl_all_gather_reduce_ms": ccdl_ms,
        "latency_ratio_ccdl_over_torch": ccdl_ms / torch_ms,
        "relative_l2": error,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
