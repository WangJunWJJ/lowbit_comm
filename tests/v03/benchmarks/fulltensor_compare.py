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


def _module() -> object:
    path = os.environ["LOWBIT_COMM_CUDA_EXTENSION_PATH"]
    spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(operation, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    local_ms = (time.perf_counter() - start) * 1000.0 / iterations
    value = torch.tensor([local_ms], device="cuda", dtype=torch.float64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


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
        backend.lower(program, context, RuntimeBindings(process_group=None))
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
    for round_index in range(3):
        order = ("native", "compressed") if round_index % 2 == 0 else (
            "compressed",
            "native",
        )
        result = {}
        for name in order:
            operation = native if name == "native" else compressed
            result[name] = _measure(operation, warmup=3, iterations=iterations)
        rounds.append(result)
    native_median = statistics.median(item["native"] for item in rounds)
    compressed_median = statistics.median(item["compressed"] for item in rounds)
    native_inplace_ms = _measure(native_inplace, warmup=3, iterations=iterations)
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
                    "native_inplace_ms": native_inplace_ms,
                    "compressed_median_ms": compressed_median,
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
