"""A6000 correctness smoke for compiled async ring/tree topology executors."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig, compile
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.registry import BackendRegistry


def _compile(*, source: torch.Tensor, rank: int, world_size: int, async_op: bool):
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=status)
    return compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8, group_size=64),
            async_op=async_op,
        ),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=str(source.device),
            shape=tuple(source.shape),
            dtype="fp16",
        ),
        registry=registry,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--mode", choices=("sync", "async", "both"), default="both")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    source = torch.linspace(0.25, 1.25, args.numel, dtype=torch.float32, device=device).half()
    source.add_(rank * 0.125)
    reference = source.clone()
    dist.all_reduce(reference)
    reference.div_(world_size)

    errors: dict[str, float] = {}
    queries: dict[str, bool] = {}
    modes = {"sync": (False,), "async": (True,), "both": (False, True)}
    for async_op in modes[args.mode]:
        executor = _compile(
            source=source,
            rank=rank,
            world_size=world_size,
            async_op=async_op,
        )
        work = executor.run(source.clone())
        queries[str(async_op).lower()] = bool(work.query())
        result = work.wait()
        if args.diagnostic:
            torch.cuda.synchronize(device)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "async_op": async_op,
                        "finite": int(torch.isfinite(result).sum()),
                        "numel": int(result.numel()),
                        "nan": int(torch.isnan(result).sum()),
                        "posinf": int(torch.isposinf(result).sum()),
                        "neginf": int(torch.isneginf(result).sum()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        relative_l2 = float(
            (result.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-12)
        )
        if not torch.isfinite(result).all():
            raise AssertionError("topology result contains non-finite values")
        if relative_l2 > 0.05:
            raise AssertionError(f"topology relative_l2 too high: {relative_l2}")
        errors[str(async_op).lower()] = relative_l2

    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": args.numel,
                    "expected_method": (
                        "ring" if world_size > 2 and args.numel % world_size == 0 else "tree"
                    ),
                    "relative_l2": errors,
                    "initial_query": queries,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
