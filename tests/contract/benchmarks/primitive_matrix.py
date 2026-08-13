"""Measure only CUDA primitives that are bound to production executors."""

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
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    FullPrecisionWire,
    FullTensor,
    NativeAllReduce,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
    RuntimeBindings,
)


def _module() -> object:
    path = os.environ["LOWBIT_COMM_CUDA_EXTENSION_PATH"]
    spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(operation, iterations: int) -> tuple[list[float], int]:
    for _ in range(5):
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


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    numel = int(os.environ.get("LOWBIT_COMM_BENCH_NUMEL", str(16 * 1024 * 1024)))
    iterations = int(os.environ.get("LOWBIT_COMM_BENCH_ITERS", "30"))
    source = torch.randn(numel, device="cuda", dtype=torch.float16) * 0.125
    context = CompileContext(
        rank=rank,
        world_size=world_size,
        shape=(numel,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
        topology_signature="single_node_pcie",
    )
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _module()))
    wire = QuantizedWire(8, 64, compact=False)
    programs = {
        "native_all_reduce": CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            FullPrecisionWire(DataType.FP16),
            NativeAllReduce(),
        ),
        "compressed_all_gather": CommunicationProgram(
            ReduceMean(), FullTensor(DataType.FP16), wire, CompressedAllGather()
        ),
        "compressed_reduce_scatter": CommunicationProgram(
            ReduceMean(),
            ReducedShard(DataType.FP16, 0),
            wire,
            CompressedReduceScatter(),
        ),
        "compressed_rs_ag": CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            wire,
            CompressedReduceScatterAllGather(),
        ),
    }
    results = {}
    for name, program in programs.items():
        lowered = backend.lower(program, context, RuntimeBindings())
        executable = backend.compile(lowered)
        samples, peak = _measure(lambda: executable.run(source).wait(), iterations)
        results[name] = {
            "physical_primitive": lowered.physical_primitive.value,
            "output": (
                "reduced_shard"
                if isinstance(program.output, ReducedShard)
                else "full_tensor"
            ),
            "p50_ms": _percentile(samples, 50),
            "p95_ms": _percentile(samples, 95),
            "median_ms": statistics.median(samples),
            "peak_memory_bytes": peak,
        }
    if rank == 0:
        print(json.dumps({"world_size": world_size, "numel": numel, "modes": results}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
