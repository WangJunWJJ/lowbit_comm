"""Rank-local torchrun worker for real FullTensor CUDA collectives."""

# ruff: noqa: E402 -- direct torchrun scripts must add the repository root.

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
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
        choices=("torch_native", "native", "int8"),
        required=True,
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--reduction", choices=("sum", "mean"), required=True)
    parser.add_argument("--numel", type=int, default=4097)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--run-index", type=int, default=0)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _benchmark_record_keys() -> frozenset[str]:
    return frozenset(
        {
            "schema_version",
            "strategy",
            "dtype",
            "reduction",
            "ranks",
            "numel",
            "group_size",
            "logical_bytes",
            "packed_bytes_per_rank",
            "warmup",
            "iterations",
            "run_index",
            "latency_ms",
            "latency_min_ms",
            "latency_max_ms",
            "effective_payload_gbps",
            "relative_l2",
            "cosine",
            "hostname",
            "gpu_name",
            "gpu_indices",
            "torch_version",
            "cuda_version",
            "container_image",
        }
    )


def _validate_benchmark_record(record: dict[str, object]) -> None:
    assert frozenset(record) == _benchmark_record_keys()
    for key in (
        "latency_ms",
        "latency_min_ms",
        "latency_max_ms",
        "effective_payload_gbps",
        "relative_l2",
        "cosine",
    ):
        value = record[key]
        assert type(value) is float
        assert math.isfinite(value)
        assert value >= 0.0
    assert record["latency_ms"] > 0.0
    assert record["latency_min_ms"] > 0.0
    assert record["latency_max_ms"] > 0.0
    for key in ("hostname", "gpu_name", "torch_version", "cuda_version"):
        assert type(record[key]) is str and record[key]


def _benchmark_intent(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    reduction: ReductionOp,
) -> CommunicationIntent:
    return CommunicationIntent(
        tensor=TensorSpec(dtype=args.dtype, shape=(args.numel,)),
        shape_family=ShapeFamily(max_numel=args.numel, alignment=1),
        reduction=reduction,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=world_size,
        rank=rank,
    )


def _benchmark_strategy(args: argparse.Namespace) -> StrategySpec:
    if args.strategy == "int8":
        return StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=args.group_size,
        )
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )


def _run_benchmark(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    reduction: ReductionOp,
) -> None:
    assert args.numel > 0
    assert args.warmup >= 0
    assert args.iterations > 0
    base = torch.linspace(
        -1.0,
        1.0,
        args.numel,
        dtype=torch.float32,
        device="cuda",
    )
    template = (base + rank * 0.125).to(_torch_dtype(args.dtype))
    expected = (
        world_size * base
        + 0.125 * world_size * (world_size - 1) / 2
    ).to(_torch_dtype(args.dtype))
    if reduction is ReductionOp.MEAN:
        expected = expected / world_size

    plan = None
    if args.strategy != "torch_native":
        plan = CudaBackend(dist.group.WORLD).lower(
            _benchmark_intent(args, rank, world_size, reduction),
            _benchmark_strategy(args),
        )

    def execute(value: torch.Tensor) -> torch.Tensor:
        if args.strategy == "torch_native":
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            if reduction is ReductionOp.MEAN:
                value.div_(world_size)
            return value
        assert plan is not None
        return plan.execute(value).wait().value

    for _ in range(args.warmup):
        execute(template.clone())
    torch.cuda.synchronize()

    latencies: list[float] = []
    actual = template.clone()
    for _ in range(args.iterations):
        actual.copy_(template)
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        actual = execute(actual)
        end.record()
        end.synchronize()
        latencies.append(float(start.elapsed_time(end)))

    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    relative_l2 = float(
        (
            (actual_fp32 - expected_fp32).norm()
            / expected_fp32.norm().clamp_min(1e-12)
        ).item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_fp32.flatten(),
            expected_fp32.flatten(),
            dim=0,
        ).item()
    )
    gathered_latencies: list[list[float] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered_latencies, latencies)
    if rank != 0:
        return

    rank_latencies = [
        latency
        for latency in gathered_latencies
        if type(latency) is list
    ]
    iteration_maxima = [
        max(rank_values[index] for rank_values in rank_latencies)
        for index in range(args.iterations)
    ]
    latency_ms = float(statistics.median(iteration_maxima))
    logical_bytes = args.numel * 2
    groups = (args.numel + args.group_size - 1) // args.group_size
    packed_bytes = (
        groups * (args.group_size + 2)
        if args.strategy == "int8"
        else logical_bytes
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "strategy": args.strategy,
        "dtype": args.dtype,
        "reduction": args.reduction,
        "ranks": world_size,
        "numel": args.numel,
        "group_size": args.group_size if args.strategy == "int8" else None,
        "logical_bytes": logical_bytes,
        "packed_bytes_per_rank": packed_bytes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "run_index": args.run_index,
        "latency_ms": latency_ms,
        "latency_min_ms": float(min(iteration_maxima)),
        "latency_max_ms": float(max(iteration_maxima)),
        "effective_payload_gbps": float(logical_bytes / latency_ms / 1e6),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_indices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda or "unknown",
        "container_image": os.environ.get("CCDL_TEST_IMAGE", "unknown"),
    }
    _validate_benchmark_record(record)
    print(f"FULLTENSOR_JSON {json.dumps(record, sort_keys=True)}", flush=True)


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
    if args.benchmark:
        _run_benchmark(args, rank, world_size, reduction)
        dist.barrier()
        dist.destroy_process_group()
        return
    if args.strategy == "torch_native":
        raise ValueError("torch_native is available only in benchmark mode")
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
    actual = work.wait().value

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
    repeated_actual = repeated_work.wait().value
    torch.testing.assert_close(
        repeated_actual,
        actual,
        rtol=5e-3 if args.strategy == "int8" else 0.0,
        atol=5e-3 if args.strategy == "int8" else 0.0,
    )
    repeated_token = repeated_work.launch_token()
    assert repeated_token.plan_id == first_token.plan_id
    rejected_launches = int(args.strategy == "int8" and args.numel > 0)
    assert repeated_token.sequence == first_token.sequence + 1 + rejected_launches
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
