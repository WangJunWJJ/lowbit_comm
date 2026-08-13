from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda import CudaDynamicAllGather, load_cuda_extension
from lowbit_comm.core import DataType, QuantizedWire


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    shape = (rank + 1, 65 + rank)
    source = (
        torch.arange(shape[0] * shape[1], device="cuda", dtype=torch.float32)
        .mul_(0.001)
        .add_(rank)
        .to(torch.float16)
        .reshape(shape)
    )
    executable = CudaDynamicAllGather(
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        layout_generation=5,
        extension_status=status,
    )
    outputs = executable.run(source).wait()
    worst = 0.0
    for source_rank, output in enumerate(outputs):
        expected_shape = (source_rank + 1, 65 + source_rank)
        expected = (
            torch.arange(
                expected_shape[0] * expected_shape[1],
                device="cuda",
                dtype=torch.float32,
            )
            .mul_(0.001)
            .add_(source_rank)
            .to(torch.float16)
            .reshape(expected_shape)
        )
        assert tuple(output.shape) == expected_shape
        worst = max(worst, float((output - expected).abs().max()))
    assert worst < 0.02, worst
    if rank == 0:
        print(
            f"DYNAMIC_ALL_GATHER_OK world_size={world_size} "
            f"max_abs_error={worst:.8f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
