"""Two/four-rank correctness smoke for the complete native CUDA protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import ccdl_comm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=256)
    return parser.parse_args()


def _error(actual: torch.Tensor, expected: float) -> float:
    return float((actual.float() - expected).abs().max().item())


def _run_protocol(*, async_op: bool, numel: int) -> dict[str, float]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    expected_sum = float(world_size * (world_size + 1) // 2)
    errors: dict[str, float] = {}

    value = torch.full((numel,), rank + 1.0, device=device)
    errors["all_reduce"] = _error(
        ccdl_comm.all_reduce(value, async_op=async_op).wait(),
        expected_sum,
    )

    value = torch.full((numel,), float(rank), device=device)
    gathered = ccdl_comm.all_gather(value, async_op=async_op).wait()
    errors["all_gather"] = max(
        _error(output, float(source))
        for source, output in enumerate(gathered)
    )

    output = torch.empty(numel, device=device)
    inputs = tuple(
        torch.full((numel,), rank + 1.0, device=device)
        for _ in range(world_size)
    )
    errors["reduce_scatter"] = _error(
        ccdl_comm.reduce_scatter(
            output,
            input_tensors=inputs,
            async_op=async_op,
        ).wait(),
        expected_sum,
    )

    inputs = tuple(
        torch.full((numel,), rank * 100.0 + destination, device=device)
        for destination in range(world_size)
    )
    outputs = tuple(torch.empty(numel, device=device) for _ in range(world_size))
    exchanged = ccdl_comm.all_to_all(
        input_tensors=inputs,
        output_tensors=outputs,
        async_op=async_op,
    ).wait()
    errors["all_to_all"] = max(
        _error(output, source * 100.0 + rank)
        for source, output in enumerate(exchanged)
    )

    value = torch.full((numel,), 7.0 if rank == 0 else -1.0, device=device)
    errors["broadcast"] = _error(
        ccdl_comm.broadcast(value, async_op=async_op).wait(),
        7.0,
    )

    value = torch.full((numel,), rank + 1.0, device=device)
    reduced = ccdl_comm.reduce(value, async_op=async_op).wait()
    errors["reduce"] = _error(reduced, expected_sum) if rank == 0 else 0.0

    value = torch.full((numel,), rank + 1.0, device=device)
    gathered = ccdl_comm.gather(value, async_op=async_op).wait()
    errors["gather"] = (
        max(
            _error(output, float(source + 1))
            for source, output in enumerate(gathered)
        )
        if rank == 0
        else 0.0
    )

    output = torch.empty(numel, device=device)
    scatter_inputs = (
        tuple(
            torch.full((numel,), 10.0 + destination, device=device)
            for destination in range(world_size)
        )
        if rank == 0
        else ()
    )
    scattered = ccdl_comm.scatter(
        output,
        scatter_list=scatter_inputs,
        async_op=async_op,
    ).wait()
    errors["scatter"] = _error(scattered, 10.0 + rank)

    assert ccdl_comm.barrier(async_op=async_op, device=str(device)).wait() is None
    errors["barrier"] = 0.0
    return errors


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in {2, 4}:
        raise RuntimeError("native_collective_smoke requires 2 or 4 ranks")

    results = {
        mode: _run_protocol(async_op=async_op, numel=args.numel)
        for mode, async_op in (("sync", False), ("async", True))
    }
    local_max = max(max(errors.values()) for errors in results.values())
    error_tensor = torch.tensor(local_max, device="cuda")
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    global_max = float(error_tensor.item())
    if global_max != 0.0:
        raise AssertionError(f"native collective max_abs_error={global_max}")

    if rank == 0:
        report = {
            "world_size": world_size,
            "backend": dist.get_backend(),
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(local_rank),
            "collectives": list(ccdl_comm.native_collectives()),
            "modes": list(results),
            "numel": args.numel,
            "max_abs_error": global_max,
            "status": "pass",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
