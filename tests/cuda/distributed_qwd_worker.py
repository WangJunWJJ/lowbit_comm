"""Distributed A6000 oracle and lifecycle worker for the private qWD plan."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import struct
import sys
import threading
import time

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lowbit_comm._C as extension  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=4097)
    parser.add_argument("--mode", choices=("qwd", "fp_refresh"), default="qwd")
    parser.add_argument(
        "--case",
        choices=("finite", "all_nan", "infinities"),
        default="finite",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--byte-oracle", action="store_true")
    parser.add_argument(
        "--fault-stage",
        choices=(
            "quant_helper",
            "transport_wait",
            "restore_helper",
            "event_record",
            "event_synchronize",
        ),
    )
    return parser.parse_args()


def _config(numel: int, rank: int, world_size: int) -> dict[str, object]:
    shard = (numel + world_size - 1) // world_size
    start = min(rank * shard, numel)
    valid = min(shard, numel - start)
    groups = (shard + 63) // 64
    payload = groups * 68
    gathered_payload = payload * world_size
    fp32_gathered = shard * world_size * 4
    return {
        "accumulation_dtype": "fp32",
        "collective": "all_gather",
        "compression": "int8",
        "dtype": "fp16",
        "fp32_gathered_bytes": fp32_gathered,
        "global_numel": numel,
        "group_size": 64,
        "groups_per_shard": groups,
        "output_bytes": shard * world_size * 2,
        "payload_bytes_per_rank": payload,
        "qwd_gathered_payload_bytes": gathered_payload,
        "rank": rank,
        "shard_numel": shard,
        "start": start,
        "valid_numel": valid,
        "workspace_bytes": max(payload + gathered_payload, fp32_gathered),
        "world_size": world_size,
    }


def _model_copy(numel: int, shard: int, world_size: int) -> torch.Tensor:
    padded = shard * world_size
    values = (
        torch.arange(padded, dtype=torch.int64)
        .remainder(37)
        .sub(18)
        .to(torch.float32)
        .div(16)
        .to(torch.float16)
    )
    if numel < padded:
        values[numel:].zero_()
    return values.cuda()


def _master_for_rank(
    *,
    case: str,
    model: torch.Tensor,
    numel: int,
    rank: int,
    shard: int,
) -> torch.Tensor:
    start = min(rank * shard, numel)
    valid = min(shard, numel - start)
    master = torch.zeros(shard, dtype=torch.float32)
    if valid:
        model_shard = model[start : start + valid].cpu().float()
        if case == "finite":
            delta = (
                torch.arange(valid, dtype=torch.int64)
                .add(start)
                .mul(7)
                .remainder(31)
                .sub(15)
                .to(torch.float32)
                .div(16)
            )
            master[:valid].copy_(model_shard + delta)
        elif case == "all_nan":
            master[:valid].fill_(float("nan"))
        else:
            master[:valid:2] = float("inf")
            master[1:valid:2] = -float("inf")
    return master.cuda()


def _rank_master_cpu(
    *, case: str, model: torch.Tensor, numel: int, rank: int, shard: int
) -> torch.Tensor:
    return _master_for_rank(
        case=case,
        model=model,
        numel=numel,
        rank=rank,
        shard=shard,
    ).cpu()


def _nonfinite_group_payload_oracle() -> bytes:
    return bytes(64) + struct.pack("<I", 0x7F800000)


def _zero_group_payload_oracle() -> bytes:
    return bytes(64) + struct.pack("<f", 1.0e-6)


def _decode_delta(delta: torch.Tensor, shard: int) -> torch.Tensor:
    padded = torch.zeros(shard, dtype=torch.float32)
    padded[: delta.numel()].copy_(delta)
    decoded = torch.empty_like(padded)
    for start in range(0, shard, 64):
        values = padded[start : start + 64]
        if not torch.isfinite(values).all():
            payload = _nonfinite_group_payload_oracle()
            scale = struct.unpack("<f", payload[64:68])[0]
            quantized = torch.tensor(
                [value if value < 128 else value - 256 for value in payload[:64]],
                dtype=torch.float32,
            )
            decoded[start : start + values.numel()].copy_(
                quantized[: values.numel()].mul(scale).div(127.0)
            )
            continue
        scale = max(float(values.abs().max()), 1.0e-6)
        quantized = (
            values.mul(torch.tensor(127.0 / scale, dtype=torch.float32))
            .round()
            .clamp(-127, 127)
        )
        decoded[start : start + values.numel()].copy_(
            quantized.mul(torch.tensor(scale, dtype=torch.float32)).div(127.0)
        )
    return decoded


def _nonfinite_payload_oracle(delta: torch.Tensor, shard: int) -> bytes:
    padded = torch.zeros(shard, dtype=torch.float32)
    padded[: delta.numel()].copy_(delta)
    payload = bytearray()
    for start in range(0, shard, 64):
        values = padded[start : start + 64]
        if torch.isfinite(values).all():
            assert values.count_nonzero().item() == 0
            payload.extend(_zero_group_payload_oracle())
        else:
            payload.extend(_nonfinite_group_payload_oracle())
    return bytes(payload)


def _assert_nonfinite_payload_bytes(
    *, master: torch.Tensor, model: torch.Tensor, valid: int
) -> None:
    delta = master.cpu() - model.cpu().float()
    expected = _nonfinite_payload_oracle(delta, master.numel())
    actual = torch.empty(len(expected), dtype=torch.uint8, device=master.device)
    ok = extension.inplace_quantize_parameter_delta(
        master,
        model,
        actual,
        valid,
        64,
        0,
        False,
        8,
        extension.Linear,
        True,
    )
    assert ok
    actual_bytes = bytes(actual.cpu().tolist())
    assert actual_bytes == expected, (actual_bytes.hex(), expected.hex())


def _oracle(
    *,
    case: str,
    mode: str,
    model: torch.Tensor,
    numel: int,
    shard: int,
    world_size: int,
) -> torch.Tensor:
    masters = [
        _rank_master_cpu(
            case=case,
            model=model,
            numel=numel,
            rank=rank,
            shard=shard,
        )
        for rank in range(world_size)
    ]
    if mode == "fp_refresh":
        return torch.cat(masters).to(torch.float16)
    pieces = []
    model_cpu = model.cpu()
    for rank, master in enumerate(masters):
        start = min(rank * shard, numel)
        valid = min(shard, numel - start)
        model_shard = model_cpu[rank * shard : (rank + 1) * shard]
        delta = master[:valid] - model_shard[:valid].float()
        pieces.append(_decode_delta(delta, shard))
    decoded = torch.cat(pieces)
    expected = model_cpu.clone()
    expected[:numel] = (expected[:numel].float() + decoded[:numel]).to(torch.float16)
    return expected


def _profile_count(profile, fragment: str) -> int:
    return sum(event.count for event in profile.key_averages() if fragment in event.key)


def _wait_eight(work, *, expect_failure: bool) -> None:
    def wait_for_state(key: str, expected) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = work._wait_latch_state_for_test()
            if state[key] == expected:
                return
            if key == "losers_arrived" and state[key] >= expected:
                return
            time.sleep(0.001)
        raise AssertionError(f"wait latch did not reach {key}={expected}")

    barrier = threading.Barrier(8)

    def wait_once():
        barrier.wait(timeout=5.0)
        try:
            return work.wait()
        except Exception as error:  # noqa: BLE001 - cross-thread native error
            return error

    work._enable_wait_latch_for_test(7)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(wait_once) for _ in range(8)]
        wait_for_state("owner_completion_blocked", True)
        wait_for_state("losers_arrived", 7)
        work._allow_completion_for_test()
        wait_for_state("terminal_publish_attempted", True)
        work._release_losers_for_test()
        results = [future.result(timeout=5.0) for future in futures]
    if expect_failure:
        assert all(isinstance(item, Exception) for item in results), results
        assert len({str(item) for item in results}) == 1
    else:
        assert all(isinstance(item, torch.Tensor) for item in results), results
        pointers = {item.data_ptr() for item in results}
        assert len(pointers) == 1, pointers
    assert work._synchronize_count_for_test() == 1


def _assert_rejected_before_side_effects(
    plan,
    master: torch.Tensor,
    model: torch.Tensor,
    mode: str,
    message: str,
) -> None:
    before = plan._side_effect_counts_for_test()
    try:
        plan.execute(master, model, mode)
    except Exception as error:  # noqa: BLE001 - native validation error
        assert message in str(error), error
    else:
        raise AssertionError(f"invalid qWD input was accepted: {message}")
    assert plan._side_effect_counts_for_test() == before


def _run_input_rejections(
    *, config: dict[str, object], master: torch.Tensor, model: torch.Tensor
) -> None:
    plan = extension._create_qwd_plan(config, dist.group.WORLD)
    invalid_cases = (
        (master.half(), model, "qwd", "master_shard dtype"),
        (master.cpu(), model, "qwd", "master_shard must be a CUDA"),
        (
            torch.empty((master.numel(), 2), device=master.device, dtype=master.dtype)[
                :, 0
            ],
            model,
            "qwd",
            "master_shard must be contiguous",
        ),
        (master[:-1], model, "qwd", "master_shard numel"),
        (master, model.float(), "qwd", "model_copy_flat dtype"),
        (master, model.cpu(), "qwd", "model_copy_flat must be a CUDA"),
        (
            master,
            torch.empty((model.numel(), 2), device=model.device, dtype=model.dtype)[
                :, 0
            ],
            "qwd",
            "model_copy_flat must be contiguous",
        ),
        (master, model[:-1], "qwd", "model_copy_flat numel"),
        (master, model, "refresh", "mode must be qwd or fp_refresh"),
    )
    for invalid_master, invalid_model, mode, message in invalid_cases:
        _assert_rejected_before_side_effects(
            plan, invalid_master, invalid_model, mode, message
        )

    alias_model = torch.empty_like(model)
    alias_master = alias_model.view(torch.float32)[: master.numel()]
    _assert_rejected_before_side_effects(
        plan, alias_master, alias_model, "qwd", "must not alias"
    )

    other_device = torch.device(
        "cuda", (master.device.index + 1) % dist.get_world_size()
    )
    _assert_rejected_before_side_effects(
        plan,
        master.to(other_device),
        model,
        "qwd",
        "same CUDA device",
    )

    gloo_group = dist.new_group(backend="gloo")
    try:
        try:
            extension._create_qwd_plan(config, gloo_group)
        except ValueError as error:
            assert "ProcessGroupNCCL" in str(error), error
        else:
            raise AssertionError("Gloo group was accepted as ProcessGroupNCCL")
    finally:
        dist.destroy_process_group(gloo_group)


def _run_lifecycle(
    *,
    config: dict[str, object],
    master: torch.Tensor,
    model: torch.Tensor,
    mode: str,
) -> None:
    plan = extension._create_qwd_plan(config, dist.group.WORLD)
    work = plan.execute(master, model, mode)
    try:
        plan.execute(master, model, mode)
    except Exception as error:  # noqa: BLE001 - native ExecutionError
        assert "workspace pool" in str(error), error
    else:
        raise AssertionError("in-flight qWD workspace was reused")
    first = work.wait()
    second = plan.execute(master, model, mode).wait()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0, equal_nan=True)

    concurrent = plan.execute(master, model, mode)
    _wait_eight(concurrent, expect_failure=False)

    failed_plan = extension._create_qwd_plan(config, dist.group.WORLD)
    failed_plan._inject_failure_for_test("event_synchronize")
    failed = failed_plan.execute(master, model, mode)
    _wait_eight(failed, expect_failure=True)

    exhausted = extension._create_qwd_plan(config, dist.group.WORLD)
    exhausted._exhaust_sequence_for_test()
    before = exhausted._side_effect_counts_for_test()
    for _ in range(3):
        try:
            exhausted.execute(master, model, mode)
        except OverflowError:
            pass
        else:
            raise AssertionError("exhausted qWD sequence was accepted")
    assert exhausted._side_effect_counts_for_test() == before

    _run_input_rejections(config=config, master=master, model=model)


def main() -> None:
    args = _parse_args()
    assert args.numel >= 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    config = _config(args.numel, rank, world_size)
    shard = int(config["shard_numel"])
    model = _model_copy(args.numel, shard, world_size)
    original = model.clone()
    master = _master_for_rank(
        case=args.case,
        model=model,
        numel=args.numel,
        rank=rank,
        shard=shard,
    )
    if args.byte_oracle:
        assert args.case != "finite"
        model_shard = model.narrow(0, rank * shard, shard)
        _assert_nonfinite_payload_bytes(
            master=master,
            model=model_shard,
            valid=int(config["valid_numel"]),
        )
    plan = extension._create_qwd_plan(config, dist.group.WORLD)

    if args.fault_stage:
        plan._inject_failure_for_test(args.fault_stage)
        expected_error = {
            "event_record": "event record",
            "event_synchronize": "event synchronize",
        }.get(args.fault_stage, args.fault_stage)
        if args.fault_stage == "event_synchronize":
            work = plan.execute(master, model, args.mode)
            try:
                work.wait()
            except Exception as error:  # noqa: BLE001
                assert expected_error in str(error), error
            else:
                raise AssertionError("injected completion failure succeeded")
        else:
            try:
                plan.execute(master, model, args.mode)
            except Exception as error:  # noqa: BLE001
                assert expected_error in str(error), error
            else:
                raise AssertionError("injected qWD failure succeeded")
        assert torch.equal(model, original), "failed qWD attempt published state"
        try:
            plan.execute(master, model, args.mode)
        except Exception as error:  # noqa: BLE001
            assert "workspace pool is quarantined" in str(error), error
        else:
            raise AssertionError("failed qWD workspace was reused")
        assert torch.equal(model, original), "quarantine retry published state"
        dist.barrier()
        if rank == 0:
            print(
                f"QWD_FAULT_OK ranks={world_size} stage={args.fault_stage}",
                flush=True,
            )
        dist.destroy_process_group()
        return

    profile = None
    if args.profile:
        profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        profile.__enter__()
    work = plan.execute(master, model, args.mode)
    actual = work.wait()
    if profile is not None:
        profile.__exit__(None, None, None)
        if rank == 0:
            collective_events = [
                (event.key, event.count)
                for event in profile.key_averages()
                if "nccl" in event.key.lower() or "allgather" in event.key.lower()
            ]
            print(f"QWD_PROFILE_COLLECTIVES {collective_events}", flush=True)
        nccl_allgathers = sum(
            event.count
            for event in profile.key_averages()
            if "nccl" in event.key.lower() and "allgather" in event.key.lower()
        )
        assert nccl_allgathers == 1, nccl_allgathers
        if args.mode == "qwd" and shard:
            assert _profile_count(profile, "aten::copy_") == 1
            assert _profile_count(profile, "quantize_parameter_delta_kernel") == 1
            assert _profile_count(profile, "dequantize_gathered_add_kernel") == 1
        if args.mode == "fp_refresh" and shard:
            assert _profile_count(profile, "aten::copy_") == 0
            assert _profile_count(profile, "qwd_refresh_cast_kernel") == 1

    expected = _oracle(
        case=args.case,
        mode=args.mode,
        model=model,
        numel=args.numel,
        shard=shard,
        world_size=world_size,
    )
    assert actual.dtype is torch.float16
    assert actual.is_contiguous()
    assert actual.numel() == shard * world_size
    if actual.numel():
        assert actual.data_ptr() != model.data_ptr()
    assert torch.equal(model, original), "qWD plan mutated caller model state"
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=0.0, atol=0.0, equal_nan=True
    )

    if args.lifecycle:
        _run_lifecycle(config=config, master=master, model=model, mode=args.mode)

    dist.barrier()
    if rank == 0:
        print(
            f"QWD_OK ranks={world_size} mode={args.mode} case={args.case} "
            f"numel={args.numel} profile={args.profile} "
            f"lifecycle={args.lifecycle}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
