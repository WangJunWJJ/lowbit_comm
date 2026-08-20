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
import pytest

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
from lowbit_comm.backends.cuda.plan import _ReducedShardWork
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
    parser.add_argument(
        "--nonfinite-tail-case",
        choices=("mixed", "all_nan", "infinities"),
    )
    parser.add_argument(
        "--fault-stage",
        choices=(
            "quant_helper",
            "transport_wait",
            "dequant_helper",
            "event_record",
            "event_synchronize",
        ),
    )
    parser.add_argument("--true-inflight-test", action="store_true")
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
        source = _source_values(
            dtype=dtype,
            global_component=global_component,
            source_rank=source_rank,
        )
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


def _global_component(numel: int) -> torch.Tensor:
    index = torch.arange(numel, dtype=torch.int64, device="cuda")
    nonlinear = index.square().remainder(251).mul(17)
    return index.mul(37).add(nonlinear).remainder(509).sub(254)


def _source_values(
    *,
    dtype: torch.dtype,
    global_component: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    return (
        global_component + 73 * (source_rank + 1)
    ).to(dtype=dtype)


def _nonfinite_source(
    value: torch.Tensor,
    case: str,
) -> torch.Tensor:
    result = value.clone()
    if case == "all_nan":
        result.fill_(float("nan"))
    elif case == "infinities":
        result[0::2] = float("inf")
        result[1::2] = -float("inf")
    else:
        assert result.numel() >= 3
        result[-3:] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")],
            dtype=result.dtype,
            device=result.device,
        )
    return result


def _nonfinite_wire_reference(
    *,
    dtype: torch.dtype,
    metadata: ReducedShardMetadata,
    packed_by_source: list[torch.Tensor],
    rank: int,
    reduction: ReductionOp,
    world_size: int,
    group_size: int,
) -> torch.Tensor:
    groups = metadata.padded_length // group_size + (
        metadata.padded_length % group_size != 0
    )
    transport = groups * group_size
    payload_bytes = groups * (group_size + 2)
    reduced = torch.zeros(transport, dtype=torch.float32, device="cuda")
    for packed in packed_by_source:
        payload = packed.reshape(world_size, payload_bytes)[rank]
        chunks = payload.reshape(groups, group_size + 2)
        scales = (
            chunks[:, :2]
            .contiguous()
            .view(dtype)
            .float()
            .reshape(groups, 1)
        )
        quantized = chunks[:, 2:].view(torch.int8).float()
        reduced.add_((quantized * (scales / 127.0)).flatten())
    if reduction is ReductionOp.MEAN:
        reduced.mul_(1.0 / world_size)
    expected = torch.zeros(
        metadata.padded_length,
        dtype=dtype,
        device="cuda",
    )
    if metadata.valid_length:
        expected[: metadata.valid_length].copy_(
            reduced[: metadata.valid_length].to(dtype)
        )
    return expected


def _run_nonfinite_tail_test(
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
    plan: object,
    rank: int,
    reduction: ReductionOp,
    value: torch.Tensor,
    world_size: int,
) -> None:
    value = _nonfinite_source(value, args.nonfinite_tail_case)
    logical = (args.numel + world_size - 1) // world_size
    groups = logical // args.group_size + (
        logical % args.group_size != 0
    )
    transport = groups * args.group_size
    packed = torch.empty(
        world_size * groups * (args.group_size + 2),
        dtype=torch.uint8,
        device="cuda",
    )
    assert torch.ops.lowbit_comm_private.shard_quantize_pack(
        value,
        packed,
        logical,
        transport,
        world_size,
        args.group_size,
    )
    packed_by_source = [torch.empty_like(packed) for _ in range(world_size)]
    dist.all_gather(packed_by_source, packed)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        result = plan.execute(value).wait()
        torch.cuda.synchronize()

    assert _profiled_launch_count(
        profile,
        "shard_quantize_pack_kernel",
    ) == 1
    assert _profiled_launch_count(
        profile,
        "shard_dequant_reduce_kernel",
    ) == 1
    expected = _nonfinite_wire_reference(
        dtype=dtype,
        metadata=result.metadata,
        packed_by_source=packed_by_source,
        rank=rank,
        reduction=reduction,
        world_size=world_size,
        group_size=args.group_size,
    )
    torch.testing.assert_close(
        result.value[: result.metadata.valid_length],
        expected[: result.metadata.valid_length],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    padding = result.value[result.metadata.valid_length :]
    assert padding.count_nonzero().item() == 0, padding
    assert torch.isfinite(padding).all(), padding
    dist.barrier()
    if rank == 0:
        print(
            "REDUCED_SHARD_NONFINITE_TAIL_OK "
            f"ranks={world_size} dtype={args.dtype} "
            f"case={args.nonfinite_tail_case}",
            flush=True,
        )


def _run_true_inflight_test(
    *,
    isolated_plan: object,
    native_plan: object,
    rank: int,
    value: torch.Tensor,
    world_size: int,
) -> None:
    native_plan._arm_dequant_gate_for_test()
    work = native_plan.execute(value)
    assert work.is_completed() is False
    with pytest.raises(ExecutionError, match="workspace pool"):
        native_plan.execute(value.clone())

    isolated_stream = torch.cuda.Stream()
    with torch.cuda.stream(isolated_stream):
        isolated = isolated_plan.execute(value.clone())
    isolated.wait()
    assert work.is_completed() is False

    work.wait()
    reused = native_plan.execute(value.clone()).wait()
    assert reused.shape == work.wait().shape
    assert native_plan._test_delay_launch_count_for_test() == 1
    assert isolated_plan._test_delay_launch_count_for_test() == 0
    dist.barrier()
    if rank == 0:
        print(
            f"REDUCED_SHARD_TRUE_INFLIGHT_OK ranks={world_size}",
            flush=True,
        )


def _run_fault_test(
    *,
    fault_stage: str,
    metadata: ReducedShardMetadata,
    native_plan: object,
    rank: int,
    value: torch.Tensor,
    world_size: int,
) -> None:
    native_plan._inject_failure_for_test(fault_stage)
    if fault_stage == "event_synchronize":
        work = _ReducedShardWork(
            native_plan.execute(value),
            metadata,
        )
        failures = []
        for operation in (work.wait, work.wait, work.result):
            with pytest.raises(
                ExecutionError,
                match="event synchronize failure",
            ) as caught:
                operation()
            failures.append(caught.value)
        assert all(failure is failures[0] for failure in failures)
    else:
        expected_failure = {
            "event_record": "event record failure",
        }.get(fault_stage, fault_stage)
        with pytest.raises(ExecutionError, match=expected_failure):
            native_plan.execute(value)
    for _ in range(3):
        with pytest.raises(ExecutionError, match="quarantined"):
            native_plan.execute(value.clone())
    dist.barrier()
    if rank == 0:
        print(
            f"REDUCED_SHARD_FAULT_OK ranks={world_size} "
            f"stage={fault_stage}",
            flush=True,
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
    global_component = _global_component(args.numel)
    value = _source_values(
        dtype=dtype,
        global_component=global_component,
        source_rank=rank,
    )
    plan = CudaBackend(dist.group.WORLD).lower(intent, strategy)
    if args.nonfinite_tail_case is not None:
        assert args.strategy == "int8"
        assert args.numel > 0 and args.numel % world_size != 0
        _run_nonfinite_tail_test(
            args=args,
            dtype=dtype,
            plan=plan,
            rank=rank,
            reduction=reduction,
            value=value,
            world_size=world_size,
        )
        dist.destroy_process_group()
        return
    if args.true_inflight_test:
        assert args.strategy == "int8" and args.numel > 0
        extension = loader.load_extension()
        native_test_plan = extension.create_reduced_shard_plan(
            _native_config(intent, strategy, plan.layout),
            dist.group.WORLD,
        )
        isolated_test_plan = extension.create_reduced_shard_plan(
            _native_config(intent, strategy, plan.layout),
            dist.group.WORLD,
        )
        _run_true_inflight_test(
            isolated_plan=isolated_test_plan,
            native_plan=native_test_plan,
            rank=rank,
            value=value,
            world_size=world_size,
        )
        dist.destroy_process_group()
        return
    if args.fault_stage is not None:
        assert args.strategy == "int8" and args.numel > 0
        extension = loader.load_extension()
        native_test_plan = extension.create_reduced_shard_plan(
            _native_config(intent, strategy, plan.layout),
            dist.group.WORLD,
        )
        _run_fault_test(
            fault_stage=args.fault_stage,
            metadata=plan.metadata,
            native_plan=native_test_plan,
            rank=rank,
            value=value,
            world_size=world_size,
        )
        dist.destroy_process_group()
        return
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
