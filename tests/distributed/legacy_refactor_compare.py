from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl.comm import qall_reduce
from ccdl.quantization import Quantizer
from ccdl_comm import CompressionConfig, compressed_all_reduce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=16_777_216)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--legacy-method", choices=("gather", "tree", "p2p", "ring"), default="gather")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
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


def benchmark(fn, *, warmup: int, repeat: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeat


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    return float((reference_f32 - candidate_f32).norm() / reference_f32.norm().clamp_min(1e-12))


def run() -> None:
    args = parse_args()
    rank, world_size, device = setup()
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    source = torch.randn(args.numel, device=device, dtype=dtype)

    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    reference /= world_size
    torch.cuda.synchronize(device)

    legacy_quantizer = Quantizer(args.group_size, -1, args.bit, 0, False, args.dtype)
    refactor_config = CompressionConfig(bit=args.bit, group_size=args.group_size)

    def legacy_once() -> torch.Tensor:
        tensor = source.clone()
        qall_reduce(tensor, op="mean", quantizer=legacy_quantizer, method=args.legacy_method, keep_self=False)
        return tensor

    def refactor_once() -> torch.Tensor:
        return compressed_all_reduce(
            source.clone(),
            config=refactor_config,
            op="mean",
            strategy="all_gather",
            dtype=args.dtype,
        )

    legacy_ms = benchmark(legacy_once, warmup=args.warmup, repeat=args.repeat, device=device)
    refactor_ms = benchmark(refactor_once, warmup=args.warmup, repeat=args.repeat, device=device)
    legacy_result = legacy_once()
    refactor_result = refactor_once()
    torch.cuda.synchronize(device)

    summary = {
        "world_size": world_size,
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "legacy_method": args.legacy_method,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "legacy_ccdl_ms": legacy_ms,
        "refactor_ccdl_ms": refactor_ms,
        "speedup_refactor_over_legacy": legacy_ms / refactor_ms,
        "legacy_relative_l2": relative_l2(reference, legacy_result),
        "refactor_relative_l2": relative_l2(reference, refactor_result),
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
