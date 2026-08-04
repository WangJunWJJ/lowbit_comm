"""A6000 correctness and latency smoke test for the compiled CUDA backend."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from ccdl_comm import (
    CommunicationPlan,
    CompileContext,
    CompressionConfig,
    compile,
    compressed_all_reduce,
)
from ccdl_comm.collectives.all_reduce import _run_compressed_all_reduce
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.registry import BackendRegistry


def _measure(operation, *, warmup: int, repeat: int, device: torch.device) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(repeat):
        operation()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        [(time.perf_counter() - started) * 1000 / repeat],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=8_388_608)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    source = torch.randn(args.numel, dtype=torch.float16, device=device)
    reference = source.clone()
    dist.all_reduce(reference)
    reference /= world_size

    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    config = CompressionConfig(bit=8, group_size=64)
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=status)
    compiled = compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=config,
            async_op=False,
        ),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=str(device),
            shape=tuple(source.shape),
            dtype="fp16",
        ),
        registry=registry,
    )

    def native() -> None:
        value = source.clone()
        dist.all_reduce(value)
        value /= world_size

    def direct() -> None:
        _run_compressed_all_reduce(
            source.clone(),
            config=config,
            strategy="all_gather",
            dtype="fp16",
        )

    def compiled_run() -> None:
        compiled.run(source.clone()).wait()

    native_ms = _measure(native, warmup=args.warmup, repeat=args.repeat, device=device)
    direct_ms = _measure(direct, warmup=args.warmup, repeat=args.repeat, device=device)
    compiled_ms = _measure(compiled_run, warmup=args.warmup, repeat=args.repeat, device=device)
    candidate_work = compiled.run(source.clone())
    if candidate_work.execution_info is not compiled.execution_info:
        raise AssertionError("work did not retain compiled execution metadata")
    candidate = candidate_work.wait()
    counter_snapshot = candidate_work.execution_counters.snapshot()
    shortcut_candidate = compressed_all_reduce(
        source.clone(),
        config=config,
        strategy="all_gather",
        dtype="fp16",
        extension_status=status,
    )
    relative_l2 = float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )
    if not torch.isfinite(candidate).all():
        raise AssertionError("compiled result contains non-finite values")
    shortcut_max_abs_difference = float(
        (candidate.float() - shortcut_candidate.float()).abs().max()
    )
    if shortcut_max_abs_difference != 0.0:
        raise AssertionError("shortcut and compiled results differ")
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": args.numel,
                    "native_ms": native_ms,
                    "direct_ccdl_ms": direct_ms,
                    "compiled_ccdl_ms": compiled_ms,
                    "compiled_vs_direct_ratio": compiled_ms / direct_ms,
                    "relative_l2": relative_l2,
                    "shortcut_max_abs_difference": shortcut_max_abs_difference,
                    "execution_counters": {
                        "run_calls": counter_snapshot.run_calls,
                        "completed_runs": counter_snapshot.completed_runs,
                        "failed_runs": counter_snapshot.failed_runs,
                        "wait_calls": counter_snapshot.wait_calls,
                        "query_calls": counter_snapshot.query_calls,
                    },
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
