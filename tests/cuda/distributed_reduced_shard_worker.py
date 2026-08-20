"""Rank-local torchrun worker for real ReducedShard CUDA collectives."""

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
from lowbit_comm.api.result import ReducedShardMetadata
from lowbit_comm.backends.cuda.backend import CudaBackend, _native_config
from lowbit_comm.backends.cuda import loader
from lowbit_comm.core.errors import ExecutionError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("native", "int8"),
        default="native",
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--reduction", choices=("sum", "mean"), required=True)
    parser.add_argument("--numel", type=int, required=True)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _reference_metadata(
    *,
    numel: int,
    rank: int,
    world_size: int,
) -> ReducedShardMetadata:
    """Mirror the Reference backend's fixed-width ownership oracle."""
    padded_length = (numel + world_size - 1) // world_size
    offset = min(rank * padded_length, numel)
    valid_length = min(padded_length, numel - offset)
    return ReducedShardMetadata(
        global_shape=(numel,),
        offset=offset,
        valid_length=valid_length,
        padded_length=padded_length,
        owner_rank=rank,
    )


def _native_fulltensor_config(
    *,
    dtype: str,
    numel: int,
    rank: int,
    reduction: str,
    world_size: int,
) -> dict[str, object]:
    return {
        "accumulation_dtype": "fp32",
        "collective": "native",
        "compression": "none",
        "dtype": dtype,
        "gathered_payload_bytes": 0,
        "group_count": 0,
        "group_size": None,
        "logical_numel": numel,
        "numel": numel,
        "output_bytes": numel * 2,
        "padded_numel": numel,
        "payload_bytes_per_rank": 0,
        "rank": rank,
        "reduction": reduction,
        "workspace_bytes": 0,
        "world_size": world_size,
    }


def _int8_fulltensor_config(
    *,
    dtype: str,
    numel: int,
    rank: int,
    reduction: str,
    world_size: int,
) -> dict[str, object]:
    group_size = 16
    group_count = (numel + group_size - 1) // group_size
    padded_numel = group_count * group_size
    payload_bytes_per_rank = padded_numel + group_count * 2
    gathered_payload_bytes = payload_bytes_per_rank * world_size
    return {
        "accumulation_dtype": "fp32",
        "collective": "compressed_all_gather_reduce",
        "compression": "int8",
        "dtype": dtype,
        "gathered_payload_bytes": gathered_payload_bytes,
        "group_count": group_count,
        "group_size": group_size,
        "logical_numel": numel,
        "numel": numel,
        "output_bytes": numel * 2,
        "padded_numel": padded_numel,
        "payload_bytes_per_rank": payload_bytes_per_rank,
        "rank": rank,
        "reduction": reduction,
        "workspace_bytes": payload_bytes_per_rank * (world_size + 1),
        "world_size": world_size,
    }


def _observe_prelaunch_sequence_exhaustion(
    native_plan: object,
    value: torch.Tensor,
) -> tuple[list[BaseException], dict[str, int]]:
    native_plan._exhaust_sequence_for_test()
    failures: list[BaseException] = []
    for _ in range(3):
        try:
            native_plan.execute(value)
        except BaseException as error:
            failures.append(error)
        else:
            raise AssertionError("exhausted sequence unexpectedly launched")
    return failures, native_plan._side_effect_counts_for_test()


def main() -> None:
    args = _parse_args()
    assert args.numel >= 0
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
        output=OutputSemantics.REDUCED_SHARD,
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
    else:
        strategy = StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_REDUCE_SCATTER,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=16,
        )
    dtype = _torch_dtype(args.dtype)
    global_component = (
        torch.arange(args.numel, dtype=torch.int64, device="cuda")
        .remainder_(16)
        .mul_(4)
    )
    value = (global_component + 64 * (rank + 1)).to(dtype=dtype)
    plan = CudaBackend(dist.group.WORLD).lower(intent, strategy)
    if args.strategy == "int8":
        try:
            plan.execute(value)
        except ExecutionError as error:
            assert "INT8 ReducedShard execution is unsupported" in str(error)
        else:
            raise AssertionError("INT8 ReducedShard execution must fail")
        dist.barrier()
        if rank == 0:
            print("REDUCED_SHARD_INT8_UNSUPPORTED", flush=True)
        dist.destroy_process_group()
        return
    work = plan.execute(value)
    result = work.wait()

    if args.dtype == "fp16" and args.reduction == "sum" and (
        args.numel == 4097
    ):
        invalid_inputs = (
            (value.to(torch.bfloat16), "dtype mismatch"),
            (value.cpu(), "must be a CUDA tensor"),
            (value[:-1], "numel mismatch"),
            (
                torch.empty(
                    (args.numel, 2), dtype=dtype, device="cuda"
                )[:, 0],
                "must be contiguous",
            ),
        )
        for invalid, message in invalid_inputs:
            try:
                plan.native_plan.execute(invalid)
            except ExecutionError as error:
                assert message in str(error), (message, error)
            else:
                raise AssertionError(f"invalid input accepted: {message}")

    metadata = result.metadata
    expected_metadata = _reference_metadata(
        numel=args.numel,
        rank=rank,
        world_size=world_size,
    )
    assert metadata == expected_metadata
    expected_full = (
        global_component * world_size
        + 64 * world_size * (world_size + 1) // 2
    ).to(dtype=dtype)
    if reduction is ReductionOp.MEAN:
        expected_full.div_(world_size)
    expected = torch.zeros(
        expected_metadata.padded_length,
        dtype=dtype,
        device="cuda",
    )
    if expected_metadata.valid_length:
        expected[: expected_metadata.valid_length].copy_(
            expected_full[
                expected_metadata.offset : expected_metadata.stop
            ]
        )
    torch.testing.assert_close(result.value, expected, rtol=0.0, atol=0.0)

    direct_work = plan.native_plan.execute(value.clone())
    direct_actual = direct_work.wait()
    torch.testing.assert_close(direct_actual, result.value, rtol=0.0, atol=0.0)
    token = direct_work.launch_token()
    assert token.plan_id > 0
    assert token.sequence == 2
    if (
        args.dtype == "fp16"
        and args.reduction == "sum"
        and args.numel == 1
    ):
        extension = loader.load_extension()
        executor_work = extension.create_cuda_executor().run(value.clone())
        fulltensor_plan = extension.create_fulltensor_plan(
            _native_fulltensor_config(
                dtype=args.dtype,
                numel=args.numel,
                rank=rank,
                reduction=args.reduction,
                world_size=world_size,
            ),
            dist.group.WORLD,
        )
        fulltensor_work = fulltensor_plan.execute(value.clone())
        fulltensor_work.wait()
        plan_ids = {
            token.plan_id,
            executor_work.launch_token().plan_id,
            fulltensor_work.launch_token().plan_id,
        }
        assert len(plan_ids) == 3, plan_ids
    if args.dtype == "fp16" and args.reduction == "sum" and (
        args.numel in (0, 4097)
    ):
        extension = loader.load_extension()
        exhausted_reduced = extension.create_reduced_shard_plan(
            _native_config(intent, strategy, plan.layout),
            dist.group.WORLD,
        )
        exhaustion_observations = [
            (
                "reduced_native",
                *_observe_prelaunch_sequence_exhaustion(
                    exhausted_reduced,
                    value.clone(),
                ),
            )
        ]
        exhausted_fulltensor = extension.create_fulltensor_plan(
            _native_fulltensor_config(
                dtype=args.dtype,
                numel=args.numel,
                rank=rank,
                reduction=args.reduction,
                world_size=world_size,
            ),
            dist.group.WORLD,
        )
        exhaustion_observations.append(
            (
                "fulltensor_native",
                *_observe_prelaunch_sequence_exhaustion(
                    exhausted_fulltensor,
                    value.clone(),
                ),
            )
        )
        exhausted_int8 = extension.create_fulltensor_plan(
            _int8_fulltensor_config(
                dtype=args.dtype,
                numel=args.numel,
                rank=rank,
                reduction=args.reduction,
                world_size=world_size,
            ),
            dist.group.WORLD,
        )
        exhaustion_observations.append(
            (
                "fulltensor_int8",
                *_observe_prelaunch_sequence_exhaustion(
                    exhausted_int8,
                    value.clone(),
                ),
            )
        )
        no_side_effects = {
            "allocation": 0,
            "workspace_acquire": 0,
            "transport_launch": 0,
            "kernel_launch": 0,
            "work_publish": 0,
        }
        nonzero_side_effects = {
            name: counts
            for name, _, counts in exhaustion_observations
            if counts != no_side_effects
        }
        assert not nonzero_side_effects, nonzero_side_effects
        for name, failures, _ in exhaustion_observations:
            assert len(failures) == 3
            assert all(type(error) is OverflowError for error in failures), (
                name,
                failures,
            )
    gathered_metadata: list[ReducedShardMetadata | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered_metadata, metadata)
    assert all(item is not None for item in gathered_metadata)
    ownership = [
        index
        for item in gathered_metadata
        if item is not None
        for index in range(item.offset, item.stop)
    ]
    assert ownership == list(range(args.numel))

    dist.barrier()
    if rank == 0:
        print(
            f"REDUCED_SHARD_OK strategy={args.strategy} dtype={args.dtype} "
            f"reduction={args.reduction} ranks={world_size} "
            f"numel={args.numel}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
