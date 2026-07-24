from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.nn.parallel import DistributedDataParallel

from ccdl_comm.ascend.codec import dequantize_tensor_cann, quantize_tensor_cann
from ccdl_comm.ascend.loader import load_cann_extension
from ccdl_comm.communication import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    torch.manual_seed(2032 + rank)
    device = torch.device("npu", local_rank)
    extension_status = load_cann_extension()
    if not extension_status.available:
        raise RuntimeError(extension_status.reason or "ccdl_cann_ops is not available")

    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    ddp_model.register_comm_hook(
        state=None,
        hook=create_ddp_comm_hook(
            CompressionConfig(bit=8, group_size=64, error_feedback=True),
            dtype="auto",
            strategy="all_gather",
            reduce="mean",
            quantize=lambda tensor, active_config: quantize_tensor_cann(
                tensor, active_config, extension_status=extension_status
            ),
            dequantize=lambda payload, shape, active_config, active_dtype: dequantize_tensor_cann(
                payload, shape, active_config, active_dtype, extension_status=extension_status
            ),
            fuse_payload=True,
        ),
    )

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(8, 16, device=device)
        targets = torch.randn(8, 4, device=device)
        loss = torch.nn.functional.mse_loss(ddp_model(inputs), targets)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

    dist.barrier()
    if rank == 0:
        print("ccdl npu cann ddp hook smoke passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
