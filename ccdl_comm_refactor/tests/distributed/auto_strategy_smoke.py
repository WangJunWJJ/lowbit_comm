from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig, compile
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.cuda.transports import compile_chunk_plan
from ccdl_comm.registry import BackendRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, required=True)
    parser.add_argument(
        "--output-layout",
        choices=("full", "shard"),
        required=True,
    )
    parser.add_argument("--expect-strategy", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def _measure(
    operation,
    *,
    warmup: int,
    repeat: int,
    device,
    torch_module,
    dist_module,
) -> float:
    for _ in range(warmup):
        operation()
    torch_module.cuda.synchronize(device)
    dist_module.barrier()
    started = time.perf_counter()
    for _ in range(repeat):
        operation()
    torch_module.cuda.synchronize(device)
    elapsed = torch_module.tensor(
        [(time.perf_counter() - started) * 1000 / repeat],
        dtype=torch_module.float64,
        device=device,
    )
    dist_module.all_reduce(elapsed, op=dist_module.ReduceOp.MAX)
    return float(elapsed.item())


def _relative_l2(candidate, reference) -> float:
    return float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )


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
    architecture = torch.cuda.get_device_name(device)
    config = CompressionConfig(bit=8, group_size=64)
    source = (
        ((torch.arange(args.numel, dtype=torch.float32, device=device) % 2048) - 1024)
        / 1024
        + rank * 0.01
    ).half()
    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.AVG)

    output_layout = args.output_layout
    collective = "all_reduce" if output_layout == "full" else "reduce_scatter"
    registry = BackendRegistry()
    extension_status = load_cuda_extension()
    register_cuda_backends(registry, extension_status=extension_status)
    compiled = compile(
        CommunicationPlan(
            collective,
            "auto",
            compression=config,
            output_layout=output_layout,
            async_op=True,
        ),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=str(device),
            shape=tuple(source.shape),
            dtype="fp16",
            device_architecture=architecture,
        ),
        registry=registry,
    )
    info = compiled.execution_info
    if info.executed_strategy != args.expect_strategy:
        raise AssertionError(
            f"expected strategy {args.expect_strategy}, got {info.executed_strategy}"
        )
    if info.fallback_used:
        raise AssertionError(f"unexpected auto fallback: {info.fallback_reason}")

    working = source.clone()
    output = None
    if output_layout == "shard":
        chunk_plan = compile_chunk_plan(
            original_numel=args.numel,
            world_size=world_size,
        )
        output = source.new_empty((chunk_plan.shard_numel,))

    def native_operation() -> None:
        working.copy_(source)
        dist.all_reduce(working, op=dist.ReduceOp.AVG)

    def auto_operation() -> None:
        working.copy_(source)
        if output is None:
            compiled.run(working).wait()
        else:
            compiled.run(working, out=output).wait()

    native_samples = [
        _measure(
            native_operation,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
            torch_module=torch,
            dist_module=dist,
        )
        for _ in range(3)
    ]
    auto_samples = [
        _measure(
            auto_operation,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
            torch_module=torch,
            dist_module=dist,
        )
        for _ in range(3)
    ]
    candidate = (
        compiled.run(source.clone()).wait()
        if output is None
        else compiled.run(source.clone(), out=output).wait()
    )
    if output_layout == "full":
        candidate_tensor = candidate
        reference_tensor = reference
    else:
        candidate_tensor = candidate.shard[: candidate.valid_numel]
        reference_tensor = reference.reshape(-1)[
            candidate.shard_offset : candidate.shard_end
        ]
    relative_l2 = _relative_l2(candidate_tensor, reference_tensor)
    if relative_l2 > 0.1:
        raise AssertionError(f"relative L2 gate failed: {relative_l2}")

    native_ms = statistics.median(native_samples)
    auto_ms = statistics.median(auto_samples)
    result = {
        "world_size": world_size,
        "numel": args.numel,
        "output_layout": output_layout,
        "device_architecture": architecture,
        "executed_strategy": info.executed_strategy,
        "fallback_used": info.fallback_used,
        "fallback_reason": info.fallback_reason,
        "selection_reason": info.details.get("strategy_selection_reason"),
        "strategy_policy_id": info.details.get("strategy_policy_id"),
        "benchmark_matched": info.details.get("strategy_benchmark_matched"),
        "native_ms": native_ms,
        "auto_ms": auto_ms,
        "speedup_vs_native_full": native_ms / auto_ms,
        "native_samples_ms": native_samples,
        "auto_samples_ms": auto_samples,
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
