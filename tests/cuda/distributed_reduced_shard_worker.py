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
    parser.add_argument("--kernel-test", action="store_true")
    parser.add_argument("--skip-launch-profile", action="store_true")
    parser.add_argument(
        "--kernel-case",
        choices=("finite", "nonfinite", "odd_alignment"),
        default="finite",
    )
    parser.add_argument(
        "--group-size",
        choices=(16, 32, 64),
        type=int,
        default=16,
    )
    parser.add_argument("--profile-plan", action="store_true")
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
    plan: object,
    value: torch.Tensor,
) -> tuple[list[BaseException], dict[str, int]]:
    plan._exhaust_sequence_for_test()
    failures: list[BaseException] = []
    for _ in range(3):
        try:
            plan.execute(value)
        except BaseException as error:
            failures.append(error)
        else:
            raise AssertionError("exhausted sequence unexpectedly launched")
    return failures, plan._side_effect_counts_for_test()


def _reference_reduction(
    *,
    dtype: torch.dtype,
    global_component: torch.Tensor,
    reduction: ReductionOp,
    world_size: int,
) -> torch.Tensor:
    reference = torch.zeros_like(global_component, dtype=torch.float32)
    for source_rank in range(world_size):
        source = (
            global_component + 64 * (source_rank + 1)
        ).to(dtype=dtype)
        reference.add_(source.float())
    if reduction is ReductionOp.MEAN:
        reference.div_(world_size)
    return reference


def _accuracy_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual_fp32 = actual.float()
    if actual.numel() == 0:
        return 0.0, 1.0
    if expected.count_nonzero().item() == 0:
        assert actual_fp32.count_nonzero().item() == 0, actual_fp32
        return 0.0, 1.0
    relative_l2 = (
        (actual_fp32 - expected).norm()
        / expected.norm().clamp_min(1.0e-12)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(),
        expected.flatten(),
        dim=0,
    )
    assert torch.isfinite(relative_l2), relative_l2
    assert torch.isfinite(cosine), cosine
    assert relative_l2.item() <= 0.005, relative_l2
    assert cosine.item() >= 0.999, cosine
    return relative_l2.item(), cosine.item()


def _profiled_launch_count(profile, kernel_name: str) -> int:
    return sum(
        event.count
        for event in profile.key_averages()
        if kernel_name in event.key
    )


def _run_shard_quantize_pack_kernel_test(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> None:
    extension = loader.load_extension()
    del extension
    logical = (args.numel + world_size - 1) // world_size
    groups = (logical + args.group_size - 1) // args.group_size
    transport = groups * args.group_size
    payload_bytes = world_size * groups * (args.group_size + 2)
    source = (
        torch.arange(args.numel, dtype=torch.int64, device="cuda")
        .mul(11)
        .remainder(37)
        .sub(18)
        .to(dtype)
    )
    if args.kernel_case == "nonfinite" and args.numel:
        source[0 :: args.group_size] = float("nan")
        source[1 :: args.group_size] = float("inf")
        source[2 :: args.group_size] = -float("inf")
    if args.kernel_case == "odd_alignment":
        packed_backing = torch.full(
            (payload_bytes + 1,),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        packed = packed_backing[1:]
        assert packed.data_ptr() % 2 == 1
        try:
            torch.ops.lowbit_comm_private.shard_quantize_pack(
                source,
                packed,
                logical,
                transport,
                world_size,
                args.group_size,
            )
        except RuntimeError as error:
            assert "packed data must be aligned" in str(error), error
        else:
            raise AssertionError("odd packed storage offset was accepted")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                f"SHARD_QUANT_PACK_REJECTED dtype={args.dtype} "
                f"destinations={world_size} case=odd_alignment "
                "launches=0",
                flush=True,
            )
        return
    packed = torch.full(
        (payload_bytes,),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    if args.skip_launch_profile:
        supported = torch.ops.lowbit_comm_private.shard_quantize_pack(
            source,
            packed,
            logical,
            transport,
            world_size,
            args.group_size,
        )
        torch.cuda.synchronize()
        launch_evidence = "not_profiled"
    else:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            supported = torch.ops.lowbit_comm_private.shard_quantize_pack(
                source,
                packed,
                logical,
                transport,
                world_size,
                args.group_size,
            )
            torch.cuda.synchronize()
        launches = sum(
            event.count
            for event in profile.key_averages()
            if "shard_quantize_pack_kernel" in event.key
        )
        assert launches == 1, launches
        launch_evidence = str(launches)
    assert supported is True
    expected_pieces = []
    source_cpu = source.cpu()
    for destination in range(world_size):
        shard_start = destination * logical
        valid = max(0, min(logical, args.numel - shard_start))
        shard = torch.zeros(transport, dtype=dtype)
        if valid:
            shard[:valid].copy_(
                source_cpu[shard_start : shard_start + valid]
            )
        for group in range(groups):
            values = shard[
                group * args.group_size : (group + 1) * args.group_size
            ]
            finite = values.isfinite()
            has_nonfinite = not finite.all().item()
            if has_nonfinite:
                scale = torch.tensor(float("inf"), dtype=dtype)
                quantized = torch.zeros(args.group_size, dtype=torch.int8)
            else:
                scale = values.abs().max()
                if scale.item() == 0.0:
                    quantized = torch.zeros(
                        args.group_size,
                        dtype=torch.int8,
                    )
                else:
                    multiplier = torch.tensor(
                        127.0 / float(scale), dtype=torch.float32
                    )
                    quantized = (
                        values.float()
                        .mul(multiplier)
                        .round()
                        .clamp(-127, 127)
                        .to(torch.int8)
                    )
            expected_pieces.append(scale.reshape(1).view(torch.uint8))
            expected_pieces.append(quantized.view(torch.uint8))
    expected = (
        torch.cat(expected_pieces)
        if expected_pieces
        else torch.empty(0, dtype=torch.uint8)
    )
    assert torch.equal(packed.cpu(), expected)
    dist.barrier()
    if rank == 0:
        print(
            f"SHARD_QUANT_PACK_OK dtype={args.dtype} "
            f"destinations={world_size} group_size={args.group_size} "
            f"numel={args.numel} launches={launch_evidence}",
            flush=True,
        )


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
            group_size=args.group_size,
        )
    dtype = _torch_dtype(args.dtype)
    if args.kernel_test:
        _run_shard_quantize_pack_kernel_test(
            args=args,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
        )
        dist.destroy_process_group()
        return
    global_component = (
        torch.arange(args.numel, dtype=torch.int64, device="cuda")
        .remainder_(16)
        .mul_(4)
    )
    value = (global_component + 64 * (rank + 1)).to(dtype=dtype)
    plan = CudaBackend(dist.group.WORLD).lower(intent, strategy)
    profile = None
    if args.profile_plan:
        profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        )
        profile.__enter__()
    work = plan.execute(value)
    if args.strategy == "int8" and args.numel > 0:
        try:
            plan.execute(value.clone())
        except ExecutionError as error:
            assert "workspace pool" in str(error), error
        else:
            raise AssertionError("in-flight ReducedShard workspace was reused")
    result = work.wait()
    if profile is not None:
        profile.__exit__(None, None, None)
        quant_launches = _profiled_launch_count(
            profile,
            "shard_quantize_pack_kernel",
        )
        dequant_launches = _profiled_launch_count(
            profile,
            "shard_dequant_reduce_kernel",
        )
        assert quant_launches == 1, quant_launches
        assert dequant_launches == 1, dequant_launches
        if rank == 0:
            print(
                "REDUCED_SHARD_LAUNCHES "
                f"ranks={world_size} quant={quant_launches} "
                f"dequant={dequant_launches}",
                flush=True,
            )

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
    expected_full = _reference_reduction(
        dtype=dtype,
        global_component=global_component,
        reduction=reduction,
        world_size=world_size,
    )
    expected_fp32 = torch.zeros(
        expected_metadata.padded_length,
        dtype=torch.float32,
        device="cuda",
    )
    if expected_metadata.valid_length:
        expected_fp32[: expected_metadata.valid_length].copy_(
            expected_full[
                expected_metadata.offset : expected_metadata.stop
            ]
        )
    assert result.value.dtype is dtype
    assert result.value.shape == (expected_metadata.padded_length,)
    assert torch.isfinite(result.value).all()
    if expected_metadata.valid_length < expected_metadata.padded_length:
        padding = result.value[expected_metadata.valid_length :]
        assert padding.count_nonzero().item() == 0, padding
    if args.strategy == "native":
        torch.testing.assert_close(
            result.value,
            expected_fp32.to(dtype),
            rtol=0.0,
            atol=0.0,
        )
        relative_l2, cosine = 0.0, 1.0
    else:
        relative_l2, cosine = _accuracy_metrics(
            result.value,
            expected_fp32,
        )

    direct_work = plan.native_plan.execute(value.clone())
    direct_actual = direct_work.wait()
    torch.testing.assert_close(
        direct_actual,
        result.value,
        rtol=0.0,
        atol=0.0,
    )
    token = direct_work.launch_token()
    assert token.plan_id > 0
    expected_sequence = (
        3 if args.strategy == "int8" and args.numel > 0 else 2
    )
    assert token.sequence == expected_sequence, token.sequence
    repeated_work = plan.native_plan.execute(value.clone())
    repeated_actual = repeated_work.wait()
    torch.testing.assert_close(
        repeated_actual,
        direct_actual,
        rtol=0.0,
        atol=0.0,
    )
    repeated_token = repeated_work.launch_token()
    assert repeated_token.plan_id == token.plan_id
    assert repeated_token.sequence == token.sequence + 1
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

    print(
        f"REDUCED_SHARD_METRIC rank={rank} dtype={args.dtype} "
        f"reduction={args.reduction} group_size={args.group_size} "
        f"numel={args.numel} relative_l2={relative_l2:.9g} "
        f"cosine={cosine:.9g}",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print(
            f"REDUCED_SHARD_OK strategy={args.strategy} dtype={args.dtype} "
            f"reduction={args.reduction} ranks={world_size} "
            f"group_size={args.group_size} numel={args.numel}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
