"""Measure only CUDA primitives that are bound to production executors."""

from __future__ import annotations

import importlib.util
import json
import os
import time

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.backends.cuda.transports import GroupedTransportRuntime
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    FullPrecisionWire,
    FullTensor,
    HierarchicalCompressed,
    NativeAllReduce,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
    RuntimeBindings,
)
from lowbit_comm.benchmarking import (
    SCHEMA_VERSION,
    runtime_fingerprint,
    summarize_samples,
)


def _module() -> object:
    path = os.environ["LOWBIT_COMM_CUDA_EXTENSION_PATH"]
    spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    round_count = int(os.environ.get("LOWBIT_COMM_BENCH_ROUNDS", "3"))
    source = torch.randn(numel, device="cuda", dtype=torch.float16) * 0.125
    context = CompileContext(
        rank=rank,
        world_size=world_size,
        shape=(numel,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
        topology_signature=os.environ.get(
            "LOWBIT_COMM_TOPOLOGY",
            "node_ids=" + ",".join("0" for _ in range(world_size)),
        ),
        node_count=int(os.environ.get("LOWBIT_COMM_NODE_COUNT", "1")),
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
    if context.node_count > 1:
        programs["hierarchical_compressed"] = CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            wire,
            HierarchicalCompressed(max_fan_in=8),
        )
    executables = {}
    metadata = {}
    for name, program in programs.items():
        runtime = (
            GroupedTransportRuntime(new_group=lambda ranks: dist.new_group(list(ranks)))
            if isinstance(program.algorithm, HierarchicalCompressed)
            else None
        )
        lowered = backend.lower(
            program,
            context,
            RuntimeBindings(backend_runtime=runtime),
        )
        executable = backend.compile(lowered)
        executables[name] = executable
        metadata[name] = {
            "physical_primitive": lowered.physical_primitive.value,
            "output": (
                "reduced_shard"
                if isinstance(program.output, ReducedShard)
                else "full_tensor"
            ),
        }
    rounds = []
    names = list(programs)
    for round_index in range(round_count):
        order = names if round_index % 2 == 0 else list(reversed(names))
        round_result = {}
        for name in order:
            samples, peak = _measure(
                lambda executable=executables[name]: executable.run(source).wait(),
                iterations,
            )
            round_result[name] = {
                **metadata[name],
                **summarize_samples(samples),
                "peak_memory_bytes": peak,
            }
        rounds.append(round_result)
    if rank == 0:
        print(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "fingerprint": runtime_fingerprint(torch, local_rank=local_rank),
            "world_size": world_size,
            "numel": numel,
            "iterations": iterations,
            "rounds": rounds,
        }))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
