from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda import (
    CudaQuantizedReceiver,
    CudaQuantizedSender,
    load_cuda_extension,
)
from lowbit_comm.core import DataType, QuantizedWire


def _expected(shape: tuple[int, ...], rank: int) -> torch.Tensor:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return (
        torch.arange(numel, device="cuda", dtype=torch.float32)
        .mul_(0.001)
        .add_(rank)
        .to(torch.float16)
        .reshape(shape)
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    wire = QuantizedWire(8, 64)
    shapes = ((65,), (3, 5, 7), (2, 17))
    worst = 0.0
    for source_rank, destination_rank, tag in ((0, 1, 41), (1, 0, 42)):
        for generation, shape in enumerate(shapes):
            if rank == source_rank:
                sender = CudaQuantizedSender(
                    peer=destination_rank,
                    tag=tag,
                    dtype=DataType.FP16,
                    wire=wire,
                    extension_status=status,
                )
                source = _expected(shape, rank)
                if generation == len(shapes) - 1:
                    sender.send(source, layout_generation=generation)
                else:
                    sender.isend(source, layout_generation=generation).wait()
            elif rank == destination_rank:
                receiver = CudaQuantizedReceiver(
                    peer=source_rank,
                    tag=tag,
                    dtype=DataType.FP16,
                    wire=wire,
                    extension_status=status,
                    device=torch.device("cuda", local_rank),
                )
                result = (
                    receiver.recv()
                    if generation == len(shapes) - 1
                    else receiver.irecv().wait()
                )
                expected = _expected(shape, source_rank)
                worst = max(worst, float((result - expected).abs().max()))
                assert result.shape == expected.shape
                assert result.dtype == torch.float16
        dist.barrier()
    assert worst < 0.02, worst
    if rank == 0:
        print(f"P2P_ORACLE_OK max_abs_error={worst:.8f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
