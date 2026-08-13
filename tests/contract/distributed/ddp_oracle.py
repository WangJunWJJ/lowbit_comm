"""Compare Native DDP and compiled low-bit DDP on a real CUDA model."""

from __future__ import annotations

import importlib.util
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lowbit_comm.adapters.ddp import GradientFeedbackState, create_ddp_hook
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


def _train(
    *,
    compressed: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    module: object,
) -> tuple[list[float], torch.Tensor]:
    torch.manual_seed(17)
    model = torch.nn.Linear(1024, 1024, bias=False, device="cuda", dtype=torch.float16)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], bucket_cap_mb=32)
    if compressed:
        program = CommunicationProgram(
            operation=ReduceMean(),
            output=FullTensor(DataType.FP16),
            wire=QuantizedWire(8, 64, compact=False),
            algorithm=CompressedReduceScatterAllGather(),
        )
        context = CompileContext(
            rank=rank,
            world_size=world_size,
            shape=(1024 * 1024,),
            dtype=DataType.FP16,
            device_type="cuda",
            device_architecture="sm86",
            topology_signature="single_node_pcie",
        )
        backend = CudaBackend(extension_status=CudaExtensionStatus(True, module))
        executable = backend.compile(
            backend.lower(program, context, RuntimeBindings(process_group=None))
        )
        hook = create_ddp_hook(
            executable,
            state=GradientFeedbackState(),
            future_factory=torch.futures.Future,
            bucket_type=dist.GradBucket,
            return_type=torch.futures.Future[torch.Tensor],
        )
        ddp.register_comm_hook(None, hook)
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    inputs = torch.randn(
        16,
        1024,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    targets = torch.zeros(16, 1024, device="cuda", dtype=torch.float16)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        output = ddp(inputs)
        loss = torch.nn.functional.mse_loss(output.float(), targets.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    return losses, model.weight.detach().clone()


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    module = _module()
    native_losses, native_weight = _train(
        compressed=False,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        module=module,
    )
    dist.barrier()
    compressed_losses, compressed_weight = _train(
        compressed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        module=module,
    )
    assert compressed_losses[-1] < compressed_losses[0]
    relative_loss_gap = abs(compressed_losses[-1] - native_losses[-1]) / native_losses[-1]
    assert relative_loss_gap <= 0.05, relative_loss_gap
    checksum = compressed_weight.float().sum()
    checksums = [torch.empty_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)
    rank_weight_gap = max(float((checksums[0] - item).abs().item()) for item in checksums)
    assert rank_weight_gap == 0.0
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "steps": len(native_losses),
                    "native_loss_start": native_losses[0],
                    "native_loss_end": native_losses[-1],
                    "compressed_loss_start": compressed_losses[0],
                    "compressed_loss_end": compressed_losses[-1],
                    "relative_loss_gap": relative_loss_gap,
                    "rank_weight_gap": rank_weight_gap,
                    "native_weight_norm": float(native_weight.float().norm().item()),
                    "compressed_weight_norm": float(compressed_weight.float().norm().item()),
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
