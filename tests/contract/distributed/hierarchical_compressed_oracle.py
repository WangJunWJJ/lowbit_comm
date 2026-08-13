from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.transports import bind_grouped_transport
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    DataType,
    FullTensor,
    HierarchicalCompressed,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("hierarchical compressed oracle requires four ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        context = CompileContext(
            rank=rank,
            world_size=world_size,
            shape=(131_072,),
            dtype=DataType.FP16,
            device_type="cuda",
            device_architecture="sm86",
            topology_signature="node_ids=0,0,1,1",
            node_count=2,
        )
        program = CommunicationProgram(
            operation=ReduceMean(),
            output=FullTensor(DataType.FP16),
            wire=QuantizedWire(8, 64, compact=False),
            algorithm=HierarchicalCompressed(max_fan_in=8),
        )
        backend = CudaBackend()
        provisional = backend.lower(program, context, RuntimeBindings())
        assert provisional.grouped_reduction is not None
        bindings = bind_grouped_transport(
            provisional.grouped_reduction,
            new_group=lambda ranks: dist.new_group(list(ranks)),
        )
        lowered = backend.lower(
            program,
            context,
            RuntimeBindings(backend_runtime=bindings),
        )
        executable = backend.compile(lowered)
        source = torch.full(
            context.shape,
            float(rank + 1),
            dtype=torch.float16,
            device="cuda",
        )
        result = executable.run(source).wait()
        expected = torch.full_like(result, 2.5)
        max_abs_error = float((result - expected).abs().max())
        rank_gap = result.clone()
        dist.broadcast(rank_gap, src=0)
        rank_gap = float((result - rank_gap).abs().max())
        assert max_abs_error <= 0.02, max_abs_error
        assert rank_gap == 0.0, rank_gap
        if rank == 0:
            print(
                "HIERARCHICAL_COMPRESSED_OK "
                f"world_size={world_size} max_abs_error={max_abs_error:.8f} "
                f"rank_gap={rank_gap:.8f}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
