"""Alternate Native NCCL and quantized FullTensor runs on the same ranks."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)
from lowbit_comm.runtime import BudgetedWorkspacePool


def _module() -> object:
    path = os.environ["LOWBIT_COMM_CUDA_EXTENSION_PATH"]
    spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], int]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    values = torch.tensor(samples, device="cuda", dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    peak = torch.tensor(
        [torch.cuda.max_memory_allocated()],
        device="cuda",
        dtype=torch.int64,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return values.cpu().tolist(), int(peak.item())


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    numel = int(os.environ.get("LOWBIT_COMM_BENCH_NUMEL", str(16 * 1024 * 1024)))
    iterations = int(os.environ.get("LOWBIT_COMM_BENCH_ITERS", "20"))
    source = torch.randn(numel, device="cuda", dtype=torch.float16) * 0.125
    native_buffer = torch.empty_like(source)
    native_inplace_buffer = source.clone()
    workspace_pool = BudgetedWorkspacePool()
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _module()))
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=False),
        algorithm=CompressedReduceScatterAllGather(),
    )
    context = CompileContext(
        rank=rank,
        world_size=world_size,
        shape=(numel,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
        topology_signature="single_node_pcie",
    )
    executable = backend.compile(
        backend.lower(
            program,
            context,
            RuntimeBindings(process_group=None, allocator=workspace_pool),
        )
    )

    def native() -> None:
        native_buffer.copy_(source)
        dist.all_reduce(native_buffer, op=dist.ReduceOp.SUM)
        native_buffer.div_(world_size)

    def native_inplace() -> None:
        dist.all_reduce(native_inplace_buffer, op=dist.ReduceOp.SUM)
        native_inplace_buffer.div_(world_size)

    def compressed() -> None:
        executable.run(source).wait()

    rounds = []
    native_samples: list[float] = []
    compressed_samples: list[float] = []
    native_peak_memory = 0
    compressed_peak_memory = 0
    executable.run(source).wait()
    allocations_after_warmup = workspace_pool.statistics().allocation_count
    for round_index in range(3):
        order = ("native", "compressed") if round_index % 2 == 0 else (
            "compressed",
            "native",
        )
        result = {}
        for name in order:
            operation = native if name == "native" else compressed
            samples, peak_memory = _measure(
                operation,
                warmup=3,
                iterations=iterations,
            )
            result[name] = statistics.median(samples)
            if name == "native":
                native_samples.extend(samples)
                native_peak_memory = max(native_peak_memory, peak_memory)
            else:
                compressed_samples.extend(samples)
                compressed_peak_memory = max(compressed_peak_memory, peak_memory)
        rounds.append(result)
    native_median = statistics.median(item["native"] for item in rounds)
    compressed_median = statistics.median(item["compressed"] for item in rounds)
    native_inplace_samples, _ = _measure(
        native_inplace,
        warmup=3,
        iterations=iterations,
    )
    native_inplace_ms = statistics.median(native_inplace_samples)
    workspace = workspace_pool.statistics()
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": numel,
                    "dtype": "fp16",
                    "iterations": iterations,
                    "rounds_ms": rounds,
                    "native_median_ms": native_median,
                    "native_p50_ms": _percentile(native_samples, 50),
                    "native_p95_ms": _percentile(native_samples, 95),
                    "native_inplace_ms": native_inplace_ms,
                    "compressed_median_ms": compressed_median,
                    "compressed_p50_ms": _percentile(compressed_samples, 50),
                    "compressed_p95_ms": _percentile(compressed_samples, 95),
                    "workspace_allocation_count": workspace.allocation_count,
                    "workspace_reuse_count": workspace.reuse_count,
                    "workspace_peak_in_use_bytes": workspace.peak_in_use_bytes,
                    "steady_state_new_allocations": (
                        workspace.allocation_count - allocations_after_warmup
                    ),
                    "native_peak_memory_bytes": native_peak_memory,
                    "compressed_peak_memory_bytes": compressed_peak_memory,
                    "speedup_percent": (
                        (native_median / compressed_median - 1.0) * 100.0
                    ),
                    "speedup_vs_native_inplace_percent": (
                        (native_inplace_ms / compressed_median - 1.0) * 100.0
                    ),
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
