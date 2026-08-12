"""A6000 oracle for contiguous compressed full-output restoration."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import torch
import torch.distributed as dist

from ccdl_comm.communication.reduce_scatter_transport import (
    make_torch_compressed_reduce_scatter_all_gather,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension


class _LegacyDistributedProxy:
    """Hide all_gather_into_tensor to reproduce the former list/cat path."""

    def __getattr__(self, name: str):
        if name == "all_gather_into_tensor":
            raise AttributeError(name)
        return getattr(dist, name)


def _measure(
    operation,
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(repeat):
        operation()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        [(time.perf_counter() - started) * 1000.0 / repeat],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.numel % world_size:
        raise ValueError("oracle numel must be divisible by world size")

    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(20260805 + rank)
    source = torch.randn(
        args.numel,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    reference = source.clone()
    dist.all_reduce(reference)
    reference /= world_size

    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason or "CCDL CUDA extension unavailable")
    output = torch.empty_like(source)
    allocations = 0

    def allocate_full_output(tensor: torch.Tensor, participants: int) -> torch.Tensor:
        nonlocal allocations
        allocations += 1
        expected = ((tensor.numel() + participants - 1) // participants) * participants
        if output.numel() != expected:
            raise AssertionError("preallocated output does not match padded gather size")
        return output

    transport = make_torch_compressed_reduce_scatter_all_gather(
        allocate_full_output_workspace=allocate_full_output,
    )
    result = None
    for _ in range(args.iterations):
        result = transport(
            source,
            config=CompressionConfig(bit=8, group_size=64),
            op="mean",
            async_op=False,
            dtype="fp16",
            extension_status=status,
        )
    torch.cuda.synchronize(device)
    assert result is output
    if result.data_ptr() != output.data_ptr():
        raise AssertionError("full restore did not preserve caller workspace storage")

    gathered = [torch.empty_like(result) for _ in range(world_size)]
    dist.all_gather(gathered, result)
    rank_max_abs = max(float((candidate - result).abs().max()) for candidate in gathered)
    relative_l2 = float(
        (result.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )
    if rank_max_abs != 0.0:
        raise AssertionError(f"full outputs differ across ranks: max_abs={rank_max_abs}")
    if not torch.isfinite(result).all():
        raise AssertionError("compressed result contains non-finite values")

    fresh_transport = make_torch_compressed_reduce_scatter_all_gather()
    legacy_proxy = _LegacyDistributedProxy()
    legacy_transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=lambda name: (
            legacy_proxy if name == "torch.distributed" else importlib.import_module(name)
        )
    )
    call_kwargs = {
        "config": CompressionConfig(bit=8, group_size=64),
        "op": "mean",
        "async_op": False,
        "dtype": "fp16",
        "extension_status": status,
    }
    legacy_ms = _measure(
        lambda: legacy_transport(source, **call_kwargs),
        device=device,
        warmup=2,
        repeat=args.iterations,
    )
    fresh_contiguous_ms = _measure(
        lambda: fresh_transport(source, **call_kwargs),
        device=device,
        warmup=2,
        repeat=args.iterations,
    )
    reused_contiguous_ms = _measure(
        lambda: transport(source, **call_kwargs),
        device=device,
        warmup=2,
        repeat=args.iterations,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "numel": args.numel,
                    "iterations": args.iterations,
                    "allocator_calls": allocations,
                    "stable_data_ptr": True,
                    "rank_max_abs": rank_max_abs,
                    "relative_l2": relative_l2,
                    "legacy_list_cat_ms": legacy_ms,
                    "fresh_contiguous_ms": fresh_contiguous_ms,
                    "reused_contiguous_ms": reused_contiguous_ms,
                    "reused_vs_legacy_speedup": legacy_ms / reused_contiguous_ms,
                    "reused_vs_fresh_speedup": fresh_contiguous_ms
                    / reused_contiguous_ms,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
