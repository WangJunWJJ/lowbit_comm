from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from ccdl_comm import (
    CommunicationPlan,
    CommunicationStage,
    CompileContext,
    CompressionConfig,
    compile,
)
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.registry import BackendRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=8_388_608)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _measure(operation, *, warmup, repeat, torch, dist, device) -> float:
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


def _median_measure(operation, *, args, torch, dist, device) -> tuple[float, list[float]]:
    samples = [
        _measure(
            operation,
            warmup=args.warmup,
            repeat=args.repeat,
            torch=torch,
            dist=dist,
            device=device,
        )
        for _ in range(3)
    ]
    return statistics.median(samples), samples


def main() -> None:
    import torch
    import torch.distributed as dist

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    config = CompressionConfig(bit=8, group_size=64)

    context = CompileContext(
        rank=rank,
        world_size=world_size,
        device=str(device),
        shape=(args.numel,),
        dtype="fp16",
        local_rank=rank,
        local_world_size=world_size,
        node_id=0,
        node_count=1,
        device_architecture=torch.cuda.get_device_name(device),
    )
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=load_cuda_extension())
    automatic = compile(
        CommunicationPlan(
            "all_reduce",
            "auto",
            compression=config,
            async_op=True,
        ),
        context,
        registry=registry,
    )
    if automatic.execution_info.executed_strategy == "hierarchical":
        raise AssertionError("single-node hierarchical must not enter auto before a speed win")

    source = (
        ((torch.arange(args.numel, dtype=torch.float32, device=device) % 2048) - 1024)
        / 1024
        + rank * 0.01
    ).half()
    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.AVG)
    working = source.clone()

    def native_operation() -> None:
        working.copy_(source)
        dist.all_reduce(working, op=dist.ReduceOp.AVG)

    def automatic_operation() -> None:
        working.copy_(source)
        automatic.run(working).wait()

    native_ms, native_samples = _median_measure(
        native_operation,
        args=args,
        torch=torch,
        dist=dist,
        device=device,
    )
    automatic_ms, automatic_samples = _median_measure(
        automatic_operation,
        args=args,
        torch=torch,
        dist=dist,
        device=device,
    )

    # Create hierarchy-specific communicators only after measuring the clean
    # non-hierarchical baseline. Extra NCCL communicators otherwise distort
    # the comparison this gate is meant to protect.
    singleton_groups = [dist.new_group([candidate]) for candidate in range(world_size)]
    local_group = dist.group.WORLD
    inter_group = singleton_groups[rank]
    stages = (
        CommunicationStage(
            "intra_reduce_scatter",
            "reduce_scatter",
            "compressed",
            compression=config,
            process_group=local_group,
            output_layout="shard",
            async_op=False,
        ),
        CommunicationStage(
            "inter_ring",
            "all_reduce",
            "topology",
            compression=config,
            process_group=inter_group,
            output_layout="shard",
            async_op=False,
        ),
        CommunicationStage(
            "restore_full",
            "all_gather",
            "native_nccl",
            process_group=local_group,
            output_layout="full",
            async_op=False,
        ),
    )
    hierarchical = compile(
        CommunicationPlan(
            "all_reduce",
            "hierarchical",
            compression=config,
            stages=stages,
            async_op=False,
        ),
        context,
        registry=registry,
    )

    def hierarchical_operation() -> None:
        working.copy_(source)
        hierarchical.run(working).wait()

    hierarchical_ms, hierarchical_samples = _median_measure(
        hierarchical_operation,
        args=args,
        torch=torch,
        dist=dist,
        device=device,
    )
    candidate = hierarchical.run(source.clone()).wait()
    relative_l2 = float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )
    if relative_l2 > 0.1:
        raise AssertionError(f"relative L2 gate failed: {relative_l2}")

    result = {
        "world_size": world_size,
        "numel": args.numel,
        "device_architecture": torch.cuda.get_device_name(device),
        "hierarchical_strategy": hierarchical.execution_info.executed_strategy,
        "hierarchical_fallback_used": hierarchical.execution_info.fallback_used,
        "auto_strategy": automatic.execution_info.executed_strategy,
        "native_ms": native_ms,
        "auto_ms": automatic_ms,
        "hierarchical_ms": hierarchical_ms,
        "speedup_vs_native": native_ms / hierarchical_ms,
        "speedup_vs_auto": automatic_ms / hierarchical_ms,
        "native_samples_ms": native_samples,
        "auto_samples_ms": automatic_samples,
        "hierarchical_samples_ms": hierarchical_samples,
        "relative_l2": relative_l2,
    }
    if rank == 0:
        serialized = json.dumps(result, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
