"""Validate the two-collective quantized FullTensor CUDA path."""

from __future__ import annotations

import importlib.util
import json
import os

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


class _CountingModule:
    def __init__(self, module: object) -> None:
        self._module = module
        self.writeback_calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._module, name)

    def inplace_dequantize_gathered(self, *args: object) -> bool:
        self.writeback_calls += 1
        return bool(self._module.inplace_dequantize_gathered(*args))


def _load_extension() -> object:
    path = os.environ["LOWBIT_COMM_CUDA_EXTENSION_PATH"]
    spec = importlib.util.spec_from_file_location("lowbit_comm_cuda_ops", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    module = _CountingModule(_load_extension())
    numel = 131_073
    wire = QuantizedWire(bit=8, group_size=64, compact=False)
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=wire,
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
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, module))
    executable = backend.compile(
        backend.lower(program, context, RuntimeBindings(process_group=None))
    )
    source = torch.randn(numel, device="cuda", dtype=torch.float16) * 0.125
    source.add_(rank * 0.03125)
    reference = source.float()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    reference.div_(world_size)

    work = executable.run(source)
    output = work.wait()
    assert module.writeback_calls == 1
    output_float = output.float()
    max_abs_error = float((output_float - reference).abs().max().item())
    assert max_abs_error <= 0.02, (rank, max_abs_error)
    gathered_outputs = [torch.empty_like(output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, output)
    rank_max_diff = max(
        float((gathered_outputs[0] - item).abs().max().item())
        for item in gathered_outputs[1:]
    ) if world_size > 1 else 0.0
    assert rank_max_diff == 0.0
    collective_stages = [
        stage for stage in executable.lowered.stages if stage.collective
    ]
    assert [stage.name for stage in collective_stages] == [
        "quantized_reduce_scatter",
        "quantized_all_gather",
    ]
    reports = [None for _ in range(world_size)]
    dist.all_gather_object(
        reports,
        {
            "max_abs_error": max_abs_error,
            "writeback_calls": module.writeback_calls,
        },
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": numel,
                    "max_abs_error": max(item["max_abs_error"] for item in reports),
                    "rank_max_diff": rank_max_diff,
                    "writeback_calls_per_rank": [
                        item["writeback_calls"] for item in reports
                    ],
                    "collectives": [stage.name for stage in collective_stages],
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
