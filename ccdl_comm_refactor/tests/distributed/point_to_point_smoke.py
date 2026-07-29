from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, iqrecv, iqsend, qrecv, qsend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("point_to_point_smoke requires exactly 2 ranks")
    device = torch.device("cuda", local_rank)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)

    torch.manual_seed(20260729)
    reference = torch.randn(args.numel, device=device, dtype=dtype)
    blocking_recv = torch.empty_like(reference)
    async_recv = torch.empty_like(reference)

    if rank == 0:
        qsend(reference, dst=1, config=config, tag=11)
        iqsend(reference, dst=1, config=config, tag=12).wait()
    else:
        qrecv(blocking_recv, src=0, config=config, dtype=args.dtype, tag=11)
        iqrecv(async_recv, src=0, config=config, dtype=args.dtype, tag=12).wait()

    blocking_error = torch.tensor(0.0, device=device)
    async_error = torch.tensor(0.0, device=device)
    if rank == 1:
        blocking_error = relative_l2(reference, blocking_recv)
        async_error = relative_l2(reference, async_recv)
    dist.broadcast(blocking_error, src=1)
    dist.broadcast(async_error, src=1)
    summary = {
        "world_size": world_size,
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "blocking_relative_l2": float(blocking_error),
        "async_relative_l2": float(async_error),
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


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return (reference.float() - candidate.float()).norm() / reference.float().norm().clamp_min(1e-12)


if __name__ == "__main__":
    main()
