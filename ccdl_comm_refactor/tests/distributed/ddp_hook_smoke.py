from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ccdl_comm.communication import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    torch.manual_seed(2026 + rank)

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).cuda()
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    ddp_model.register_comm_hook(
        state=None,
        hook=create_ddp_comm_hook(
            CompressionConfig(bit=8, group_size=64, error_feedback=True, target="ddp_gradient_bucket"),
            dtype="auto",
            strategy="all_gather",
            reduce="mean",
        ),
    )

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(8, 16, device="cuda")
        targets = torch.randn(8, 4, device="cuda")
        loss = torch.nn.functional.mse_loss(ddp_model(inputs), targets)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

    dist.barrier()
    if rank == 0:
        print("ccdl ddp hook smoke passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
