"""A6000 latency comparison for Task 13 topology executors."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig, compile
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.cuda.transports import (
    PipelinedRingExecutor,
    TreeExecutor,
    compile_chunk_plan,
    compile_pipelined_ring_schedule,
    compile_tree_schedule,
)
from ccdl_comm.cuda.transports.torch_topology import (
    TorchPipelinedRingRuntime,
    TorchTreeRuntime,
)
from ccdl_comm.cuda.workspace import CudaShardWorkspaceProvider, create_torch_workspace_pool
from ccdl_comm.registry import BackendRegistry


def _measure(
    operation: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    rounds: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    for _ in range(rounds):
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
        samples.append(float(elapsed.item()))
    return statistics.median(samples), samples


def _compile_executor(
    strategy: str,
    *,
    source: torch.Tensor,
    rank: int,
    world_size: int,
    status: object,
    config: CompressionConfig,
):
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=status)
    return compile(
        CommunicationPlan(
            "all_reduce" if strategy != "compressed" else "reduce_scatter",
            strategy,
            compression=config,
            output_layout="full" if strategy != "compressed" else "shard",
            async_op=True,
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


def _build_topology_executor(
    method: str,
    *,
    source: torch.Tensor,
    rank: int,
    world_size: int,
    status: object,
    config: CompressionConfig,
):
    manager = CudaCompletionManager(extension_status=status)
    pool = create_torch_workspace_pool(max_entries=64)
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="cuda",
        collective="all_reduce",
        strategy=f"topology_{method}",
        device=str(source.device),
        pool_reduced_output=False,
    )
    plan = compile_chunk_plan(original_numel=source.numel(), world_size=world_size)
    runtime_type = TorchPipelinedRingRuntime if method == "ring" else TorchTreeRuntime
    runtime = runtime_type(
        config=config,
        dtype="fp16",
        world_size=world_size,
        rank=rank,
        extension_status=status,
        completion_manager=manager,
    )
    common = {
        "runtime": runtime,
        "workspace_session_factory": lambda _tensor: provider.begin(stream=runtime.stream),
        "completion_manager": manager,
    }
    if method == "ring":
        return PipelinedRingExecutor(
            schedule=compile_pipelined_ring_schedule(chunk_plan=plan, rank=rank),
            **common,
        )
    return TreeExecutor(
        schedule=compile_tree_schedule(chunk_plan=plan, rank=rank),
        **common,
    )


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=8_388_608)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    source = (
        ((torch.arange(args.numel, dtype=torch.float32, device=device) % 2048) - 1024)
        / 1024
        + rank * 0.01
    ).half()
    reference = source.clone()
    dist.all_reduce(reference)
    reference.div_(world_size)
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    config = CompressionConfig(bit=8, group_size=64)

    all_gather = _compile_executor(
        "all_gather",
        source=source,
        rank=rank,
        world_size=world_size,
        status=status,
        config=config,
    )
    reduced = _compile_executor(
        "compressed",
        source=source,
        rank=rank,
        world_size=world_size,
        status=status,
        config=config,
    )
    ring = _build_topology_executor(
        "ring",
        source=source,
        rank=rank,
        world_size=world_size,
        status=status,
        config=config,
    )
    tree = _build_topology_executor(
        "tree",
        source=source,
        rank=rank,
        world_size=world_size,
        status=status,
        config=config,
    )
    working = source.clone()
    reduced_chunk_plan = compile_chunk_plan(
        original_numel=source.numel(),
        world_size=world_size,
    )
    reduced_output = source.new_empty((reduced_chunk_plan.shard_numel,))

    def native_op() -> None:
        working.copy_(source)
        dist.all_reduce(working)
        working.div_(world_size)

    def compiled_op(executor: object, *, out: torch.Tensor | None = None) -> Callable[[], None]:
        def run() -> None:
            working.copy_(source)
            if out is None:
                executor.run(working).wait()
            else:
                executor.run(working, out=out).wait()

        return run

    operations = {
        "native_fp16": native_op,
        "compressed_all_gather": compiled_op(all_gather),
        "pipelined_ring": compiled_op(ring),
        "tree": compiled_op(tree),
        "compressed_reduced_shard": compiled_op(reduced, out=reduced_output),
    }
    timings: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for name, operation in operations.items():
        timings[name], samples[name] = _measure(
            operation,
            warmup=args.warmup,
            repeat=args.repeat,
            rounds=args.rounds,
            device=device,
        )

    accuracy: dict[str, float] = {}
    for name, executor in (("compressed_all_gather", all_gather), ("pipelined_ring", ring), ("tree", tree)):
        candidate = executor.run(source.clone()).wait()
        accuracy[name] = _relative_l2(candidate, reference)
    reduced_result = reduced.run(source.clone(), out=reduced_output).wait()
    shard_reference = reference.reshape(-1)[
        reduced_result.shard_offset : reduced_result.shard_end
    ]
    accuracy["compressed_reduced_shard"] = _relative_l2(
        reduced_result.shard[: reduced_result.valid_numel],
        shard_reference,
    )
    if any(value > 0.1 for value in accuracy.values()):
        raise AssertionError(f"topology accuracy gate failed: {accuracy}")

    if rank == 0:
        native_ms = timings["native_fp16"]
        result = {
            "world_size": world_size,
            "numel": args.numel,
            "dtype": "fp16",
            "compression": {"bit": config.bit, "group_size": config.group_size},
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "rounds": args.rounds,
            "median_ms": timings,
            "samples_ms": samples,
            "speedup_vs_native": {
                name: native_ms / value for name, value in timings.items()
            },
            "relative_l2": accuracy,
        }
        serialized = json.dumps(result, sort_keys=True)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
