"""Rank-local torchrun worker for real FullTensor CUDA collectives."""

# ruff: noqa: E402 -- direct torchrun scripts must add the repository root.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.backends.cuda.backend import CudaBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("native",), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--reduction", choices=("sum", "mean"), required=True)
    parser.add_argument("--numel", type=int, default=4097)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def main() -> None:
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    reduction = (
        ReductionOp.SUM if args.reduction == "sum" else ReductionOp.MEAN
    )
    intent = CommunicationIntent(
        tensor=TensorSpec(dtype=args.dtype, shape=(args.numel,)),
        shape_family=ShapeFamily(max_numel=args.numel, alignment=1),
        reduction=reduction,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=world_size,
        rank=rank,
    )
    strategy = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    value = torch.full(
        (args.numel,),
        rank + 1,
        dtype=_torch_dtype(args.dtype),
        device="cuda",
    )
    plan = CudaBackend(dist.group.WORLD).lower(intent, strategy)
    work = plan.execute(value)
    actual = work.wait()

    expected_scalar = world_size * (world_size + 1) / 2
    if reduction is ReductionOp.MEAN:
        expected_scalar /= world_size
    expected = torch.full_like(actual, expected_scalar)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    first_token = work.launch_token()
    assert first_token.plan_id > 0
    assert first_token.sequence > 0

    repeated = torch.full_like(actual, rank + 2)
    repeated_work = plan.execute(repeated)
    repeated_actual = repeated_work.wait()
    repeated_scalar = world_size * (world_size + 3) / 2
    if reduction is ReductionOp.MEAN:
        repeated_scalar /= world_size
    torch.testing.assert_close(
        repeated_actual,
        torch.full_like(repeated_actual, repeated_scalar),
        rtol=0.0,
        atol=0.0,
    )
    repeated_token = repeated_work.launch_token()
    assert repeated_token.plan_id == first_token.plan_id
    assert repeated_token.sequence == first_token.sequence + 1
    dist.barrier()
    if rank == 0:
        print(
            f"FULLTENSOR_OK strategy={args.strategy} dtype={args.dtype} "
            f"reduction={args.reduction} ranks={world_size}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
