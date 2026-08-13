from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda import CudaBackend, load_cuda_extension
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    DataType,
    FullPrecisionWire,
    FullTensor,
    NativeAllReduce,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    shape = (131_073,)
    context = CompileContext(
        rank=rank,
        world_size=world_size,
        shape=shape,
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
    )
    bindings = RuntimeBindings()
    source = (
        torch.arange(shape[0], device="cuda", dtype=torch.float32)
        .remainder_(997)
        .div_(997)
        .add_(rank)
        .to(torch.float16)
    )
    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    reference.div_(world_size)
    backend = CudaBackend(extension_status=load_cuda_extension())
    native_program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=FullPrecisionWire(DataType.FP16),
        algorithm=NativeAllReduce(),
    )
    native = backend.compile(backend.lower(native_program, context, bindings))
    native_output = native.run(source.clone()).wait()
    native_error = float((native_output - reference).abs().max())
    compressed_program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64),
        algorithm=CompressedAllGather(),
    )
    compressed = backend.compile(backend.lower(compressed_program, context, bindings))
    compressed_output = compressed.run(source.clone()).wait()
    compressed_error = float((compressed_output - reference).abs().max())
    rank_min = compressed_output.clone()
    rank_max = compressed_output.clone()
    dist.all_reduce(rank_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    rank_gap = float((rank_max - rank_min).abs().max())
    assert native_error < 0.002, native_error
    assert compressed_error < 0.03, compressed_error
    assert rank_gap == 0.0, rank_gap
    if rank == 0:
        print(
            f"COMPILED_MODES_OK world_size={world_size} native_error={native_error:.8f} "
            f"compressed_error={compressed_error:.8f} rank_gap={rank_gap:.8f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
