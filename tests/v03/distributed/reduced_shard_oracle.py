"""Validate quantized ReducedShard semantics on real distributed CUDA ranks."""

from __future__ import annotations

import json
import importlib.util
import os

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.build import CSRC_ROOT, ensure_generated_sources
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatter,
    DataType,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
    RuntimeBindings,
)


def _load_extension() -> object:
    prebuilt = os.environ.get("LOWBIT_COMM_CUDA_EXTENSION_PATH")
    if prebuilt:
        spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", prebuilt)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load CUDA extension from {prebuilt}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    quantization = CSRC_ROOT / "quantization"
    ensure_generated_sources(quantization)
    sources = [CSRC_ROOT / "pybind.cpp"]
    sources.extend((CSRC_ROOT / "executor").glob("*.cpp"))
    sources.extend(quantization.glob("*.cu"))
    return load(
        name="lowbit_comm_cuda_ops",
        sources=sorted(str(path) for path in sources),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__"],
        verbose=False,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    module = _load_extension()
    numel = 131_073
    wire = QuantizedWire(bit=8, group_size=64)
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=ReducedShard(DataType.FP16, layout_version=1),
        wire=wire,
        algorithm=CompressedReduceScatter(),
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
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, module))
    executable = backend.compile(
        backend.lower(program, context, RuntimeBindings(process_group=None))
    )
    source = torch.linspace(-1.0, 1.0, numel, device="cuda", dtype=torch.float32)
    source = source.add(rank * 0.125).to(torch.float16)
    assert bool(torch.isfinite(source).all()), "oracle input must be finite"
    full_reference = source.clone()
    dist.all_reduce(full_reference, op=dist.ReduceOp.SUM)
    full_reference.div_(world_size)

    result = executable.run(source).wait()
    torch.cuda.synchronize()
    start, end = result.logical_range
    expected = full_reference[start:end]
    actual = result.tensor[: result.valid_numel]
    max_abs_error = (
        float((actual - expected).abs().max().item()) if result.valid_numel else 0.0
    )
    assert max_abs_error <= 0.02, (rank, max_abs_error)
    assert not any("all_gather" in stage.name for stage in executable.lowered.stages)

    errors = [None for _ in range(world_size)]
    dist.all_gather_object(errors, max_abs_error)
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": numel,
                    "shard_numel": executable.plan.shard_numel,
                    "max_abs_error": max(errors),
                    "stages": [stage.name for stage in executable.lowered.stages],
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
