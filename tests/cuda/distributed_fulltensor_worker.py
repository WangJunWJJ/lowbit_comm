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
from lowbit_comm.core.errors import ExecutionError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("native", "int8"),
        required=True,
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--reduction", choices=("sum", "mean"), required=True)
    parser.add_argument("--numel", type=int, default=4097)
    parser.add_argument("--group-size", type=int, default=16)
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
    if args.strategy == "native":
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
        expected_scalar = world_size * (world_size + 1) / 2
        expected = torch.full_like(value, expected_scalar)
    else:
        strategy = StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=args.group_size,
        )
        base = torch.linspace(
            -1.0,
            1.0,
            args.numel,
            dtype=torch.float32,
            device="cuda",
        )
        value = (base + rank * 0.125).to(_torch_dtype(args.dtype))
        expected = (
            world_size * base
            + 0.125 * world_size * (world_size - 1) / 2
        ).to(_torch_dtype(args.dtype))
    repeated_input = value.clone()
    plan = CudaBackend(dist.group.WORLD).lower(intent, strategy)
    work = plan.execute(value)
    if args.strategy == "int8" and args.numel > 0:
        try:
            plan.execute(repeated_input.clone())
        except ExecutionError as error:
            assert "workspace pool" in str(error)
        else:
            raise AssertionError("in-flight workspace was reused")
    actual = work.wait()

    if reduction is ReductionOp.MEAN:
        expected = expected / world_size
    if args.strategy == "native":
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    else:
        actual_fp32 = actual.float()
        expected_fp32 = expected.float()
        if actual.numel() == 0:
            relative_l2 = torch.zeros((), device="cuda")
            cosine = torch.ones((), device="cuda")
        else:
            relative_l2 = (
                (actual_fp32 - expected_fp32).norm()
                / expected_fp32.norm().clamp_min(1e-12)
            )
            cosine = torch.nn.functional.cosine_similarity(
                actual_fp32.flatten(),
                expected_fp32.flatten(),
                dim=0,
            )
        assert torch.isfinite(relative_l2)
        assert torch.isfinite(cosine)
        assert cosine.item() > 0.99
    first_token = work.launch_token()
    assert first_token.plan_id > 0
    assert first_token.sequence > 0

    repeated_work = plan.execute(repeated_input)
    repeated_actual = repeated_work.wait()
    torch.testing.assert_close(
        repeated_actual,
        actual,
        rtol=5e-3 if args.strategy == "int8" else 0.0,
        atol=5e-3 if args.strategy == "int8" else 0.0,
    )
    repeated_token = repeated_work.launch_token()
    assert repeated_token.plan_id == first_token.plan_id
    assert repeated_token.sequence == first_token.sequence + 1
    dist.barrier()
    if rank == 0:
        metrics = ""
        if args.strategy == "int8":
            metrics = (
                f" relative_l2={relative_l2.item():.8f}"
                f" cosine={cosine.item():.8f}"
            )
        print(
            f"FULLTENSOR_OK strategy={args.strategy} dtype={args.dtype} "
            f"reduction={args.reduction} ranks={world_size}{metrics}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
