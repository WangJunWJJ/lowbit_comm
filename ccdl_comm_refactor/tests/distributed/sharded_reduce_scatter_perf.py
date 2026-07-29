from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_reduce_scatter_shard
from ccdl_comm.communication.reduce_scatter_transport import make_torch_compressed_reduce_scatter_shard


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

    shard_transport = make_torch_compressed_reduce_scatter_shard()
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

    torch_ms = benchmark(torch_reduce_scatter_once, warmup=args.warmup, repeat=args.repeat, device=device)
    ccdl_ms = benchmark(ccdl_shard_once, warmup=args.warmup, repeat=args.repeat, device=device)
    ccdl_result = ccdl_shard_once()
    torch.cuda.synchronize(device)
    error = relative_l2(reference_shard, ccdl_result)

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
        "torch_reduce_scatter_ms": torch_ms,
        "ccdl_shard_ms": ccdl_ms,
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
