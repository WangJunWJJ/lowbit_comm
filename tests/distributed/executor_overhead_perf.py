"""Measure compiled-plan dispatch overhead on a real CUDA collective."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import (
    CommunicationPlan,
    CompileCache,
    CompileContext,
    CompressionConfig,
    compile as compile_plan,
)
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.registry import BackendRegistry
from tests.benchmarks.result_schema import resolve_benchmark_identity


def positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse reproducible executor-overhead benchmark arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=positive_integer, default=1000)
    parser.add_argument("--warmup", type=positive_integer, default=100)
    parser.add_argument("--numel", type=positive_integer, default=8_388_608)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def measure_cuda_us(
    operation,
    tensor: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Measure one callable with CUDA events and host wall-clock time."""

    for _ in range(warmup):
        operation(tensor)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        operation(tensor)
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter_ns() - wall_started) / 1000.0 / iterations
    device_us = start.elapsed_time(end) * 1000.0 / iterations
    return device_us, wall_us


def maximum_rank_value(value: float, device: torch.device) -> float:
    """Return the maximum scalar reported by any participating rank."""

    measurement = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(measurement, op=dist.ReduceOp.MAX)
    return float(measurement.item())


def main() -> None:
    """Compile once, measure direct/exposed dispatch, and write rank-zero JSON."""

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    tensor = torch.randn(args.numel, dtype=torch.float16, device=device)

    extension_status = load_cuda_extension()
    if not extension_status.available:
        raise RuntimeError(extension_status.reason)
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=extension_status)
    cache = CompileCache(max_entries=4)
    plan = CommunicationPlan(
        "all_reduce",
        "all_gather",
        compression=CompressionConfig(bit=8, group_size=64),
        async_op=False,
    )
    context = CompileContext(
        rank=rank,
        world_size=world_size,
        device=str(device),
        shape=tuple(tensor.shape),
        dtype="fp16",
    )

    compile_started = time.perf_counter_ns()
    compiled = compile_plan(plan, context, registry=registry, cache=cache)
    compile_us = maximum_rank_value(
        (time.perf_counter_ns() - compile_started) / 1000.0,
        device,
    )
    cached = compile_plan(plan, context, registry=registry, cache=cache)
    cache_hit = cached is compiled and len(cache) == 1
    if not cache_hit:
        raise RuntimeError("compiled plan cache did not return the original plan")

    direct_first_us, wall_direct_first_us = measure_cuda_us(
        compiled.executor.run,
        tensor,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    compiled_second_us, wall_compiled_second_us = measure_cuda_us(
        compiled.run,
        tensor,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    compiled_first_us, wall_compiled_first_us = measure_cuda_us(
        compiled.run,
        tensor,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    direct_second_us, wall_direct_second_us = measure_cuda_us(
        compiled.executor.run,
        tensor,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    local_direct_samples_us = [direct_first_us, direct_second_us]
    local_compiled_samples_us = [compiled_first_us, compiled_second_us]
    local_wall_direct_samples_us = [wall_direct_first_us, wall_direct_second_us]
    local_wall_compiled_samples_us = [wall_compiled_first_us, wall_compiled_second_us]
    local_direct_us = sum(local_direct_samples_us) / len(local_direct_samples_us)
    local_compiled_us = sum(local_compiled_samples_us) / len(local_compiled_samples_us)
    local_wall_direct_us = sum(local_wall_direct_samples_us) / len(local_wall_direct_samples_us)
    local_wall_compiled_us = sum(local_wall_compiled_samples_us) / len(local_wall_compiled_samples_us)
    local_device_ratio = local_compiled_us / local_direct_us
    local_wall_ratio = local_wall_compiled_us / local_wall_direct_us
    local_overhead_ratio = max(local_device_ratio, local_wall_ratio)

    direct_samples_us = [maximum_rank_value(value, device) for value in local_direct_samples_us]
    compiled_samples_us = [maximum_rank_value(value, device) for value in local_compiled_samples_us]
    wall_direct_samples_us = [maximum_rank_value(value, device) for value in local_wall_direct_samples_us]
    wall_compiled_samples_us = [maximum_rank_value(value, device) for value in local_wall_compiled_samples_us]
    direct_executor_us = maximum_rank_value(local_direct_us, device)
    compiled_plan_us = maximum_rank_value(local_compiled_us, device)
    wall_direct_executor_us = maximum_rank_value(local_wall_direct_us, device)
    wall_compiled_plan_us = maximum_rank_value(local_wall_compiled_us, device)
    device_overhead_ratio = maximum_rank_value(local_device_ratio, device)
    wall_overhead_ratio = maximum_rank_value(local_wall_ratio, device)
    overhead_ratio = maximum_rank_value(local_overhead_ratio, device)

    if rank == 0:
        identity = resolve_benchmark_identity()
        result = {
            **identity,
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_version": str(torch.version.cuda),
            "torch_version": torch.__version__,
            "world_size": world_size,
            "dtype": "fp16",
            "numel": args.numel,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "direct_executor_us": direct_executor_us,
            "compiled_plan_us": compiled_plan_us,
            "overhead_ratio": overhead_ratio,
            "device_overhead_ratio": device_overhead_ratio,
            "wall_overhead_ratio": wall_overhead_ratio,
            "wall_direct_executor_us": wall_direct_executor_us,
            "wall_compiled_plan_us": wall_compiled_plan_us,
            "compile_us": compile_us,
            "cache_hit": cache_hit,
            "direct_samples_us": direct_samples_us,
            "compiled_samples_us": compiled_samples_us,
            "wall_direct_samples_us": wall_direct_samples_us,
            "wall_compiled_samples_us": wall_compiled_samples_us,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
