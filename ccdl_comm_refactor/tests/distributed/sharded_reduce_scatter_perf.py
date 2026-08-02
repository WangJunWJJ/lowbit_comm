from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_reduce_scatter_shard
from ccdl_comm.communication import make_native_topology_reduce_scatter_shard
from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
from tests.benchmarks.result_schema import resolve_benchmark_identity, validate_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=16_777_216)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--transport", choices=("compressed", "topology"), default="compressed")
    parser.add_argument("--topology-method", choices=("auto", "p2p", "ring"), default="auto")
    return parser.parse_args()


def setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def benchmark(fn, *, warmup: int, repeat: int, device: torch.device) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000 / repeat
    peak_memory = torch.cuda.max_memory_allocated(device)
    measurements = torch.tensor([latency_ms, float(peak_memory)], dtype=torch.float64, device=device)
    dist.all_reduce(measurements, op=dist.ReduceOp.MAX)
    return float(measurements[0].item()), int(measurements[1].item())


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    return float((reference_f32 - candidate_f32).norm() / reference_f32.norm().clamp_min(1e-12))


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = reference_f32 - candidate_f32
    return {
        "relative_l2": float(difference.norm() / reference_f32.norm().clamp_min(1e-12)),
        "max_abs_error": float(difference.abs().max()),
        "rmse": float(difference.square().mean().sqrt()),
        "non_finite": int((~torch.isfinite(candidate_f32)).sum().item()),
    }


def result_record(
    *,
    strategy: str,
    latency_ms: float,
    peak_memory_bytes: int,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    args: argparse.Namespace,
    world_size: int,
    device: torch.device,
    identity: dict[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {
        **identity,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": str(torch.version.cuda),
        "torch_version": torch.__version__,
        "world_size": world_size,
        "dtype": args.dtype,
        "numel": args.numel,
        "strategy": strategy,
        "latency_ms": latency_ms,
        "effective_gbps": args.numel * candidate.element_size() / latency_ms / 1_000_000,
        "peak_memory_bytes": peak_memory_bytes,
        **error_metrics(reference, candidate),
    }
    validate_result(result)
    return result


def run() -> None:
    args = parse_args()
    rank, world_size, device = setup()
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    identity = resolve_benchmark_identity()
    source = torch.randn(args.numel, device=device, dtype=dtype)
    shard_numel = (args.numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    if padded_numel != args.numel:
        source_for_reference = torch.cat((source, source.new_zeros((padded_numel - args.numel,))), dim=0)
    else:
        source_for_reference = source

    full_reference = source_for_reference.clone()
    dist.all_reduce(full_reference, op=dist.ReduceOp.SUM)
    full_reference /= world_size
    reference_shard = full_reference.narrow(0, rank * shard_numel, shard_numel).contiguous()
    torch.cuda.synchronize(device)

    shard_transport = (
        make_native_topology_reduce_scatter_shard(
            method=None if args.topology_method == "auto" else args.topology_method
        )
        if args.transport == "topology"
        else None
    )
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)
    compiled_plan = (
        compile_cuda_shortcut(
            source,
            collective="reduce_scatter",
            strategy="compressed",
            output_layout="shard",
            config=config,
            async_op=False,
            dtype=args.dtype,
            extension_status=None,
        )
        if args.transport == "compressed"
        else None
    )

    def compressed_full_restore_once() -> torch.Tensor:
        reduced = ccdl_reduced_shard_once()
        restored_shards = [reduced.shard.new_empty((reduced.shard_numel,)) for _ in range(world_size)]
        dist.all_gather(restored_shards, reduced.shard)
        flattened = torch.cat(restored_shards, dim=0)
        return flattened.narrow(0, rank * shard_numel, shard_numel).contiguous()

    def ccdl_shard_once() -> torch.Tensor:
        return ccdl_reduced_shard_once().shard

    def ccdl_reduced_shard_once():
        if compiled_plan is not None:
            return compiled_plan.run(source).wait()
        return compressed_reduce_scatter_shard(
            source,
            config=config,
            op="mean",
            dtype=args.dtype,
            reduce_scatter_shard=shard_transport,
        )

    full_first_ms, full_first_peak = benchmark(
        compressed_full_restore_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    shard_first_ms, shard_first_peak = benchmark(
        ccdl_shard_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    shard_second_ms, shard_second_peak = benchmark(
        ccdl_shard_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    full_second_ms, full_second_peak = benchmark(
        compressed_full_restore_once, warmup=args.warmup, repeat=args.repeat, device=device
    )
    full_restore_ms = (full_first_ms + full_second_ms) / 2.0
    ccdl_ms = (shard_first_ms + shard_second_ms) / 2.0
    full_restore_peak = max(full_first_peak, full_second_peak)
    ccdl_peak = max(shard_first_peak, shard_second_peak)
    full_restore_result = compressed_full_restore_once()
    reduced_shard = ccdl_reduced_shard_once()
    ccdl_result = reduced_shard.shard
    torch.cuda.synchronize(device)
    error = relative_l2(reference_shard, ccdl_result)
    if error > 0.02:
        raise AssertionError(f"relative L2 {error} exceeds approved INT8 threshold 0.02")
    local_metadata = reduced_shard.to_metadata()
    if args.transport == "compressed":
        metadata = local_metadata["metadata"]
        assert metadata["chunk_plan_precompiled"] is True
        assert metadata["workspace_cache"] is True
    rank_metadata = [None for _ in range(world_size)]
    dist.all_gather_object(rank_metadata, local_metadata)
    invariant_keys = ("original_shape", "original_numel", "padded_numel", "world_size", "reduce", "dtype")
    for metadata in rank_metadata:
        assert metadata is not None
        assert all(metadata[key] == local_metadata[key] for key in invariant_keys)
    assert tuple(metadata["shard_index"] for metadata in rank_metadata) == tuple(range(world_size))
    results = [
        result_record(
            strategy="ccdl_compressed_full_restore",
            latency_ms=full_restore_ms,
            peak_memory_bytes=full_restore_peak,
            reference=reference_shard,
            candidate=full_restore_result,
            args=args,
            world_size=world_size,
            device=device,
            identity=identity,
        ),
        result_record(
            strategy=f"ccdl_compressed_reduce_scatter_{args.transport}",
            latency_ms=ccdl_ms,
            peak_memory_bytes=ccdl_peak,
            reference=reference_shard,
            candidate=ccdl_result,
            args=args,
            world_size=world_size,
            device=device,
            identity=identity,
        ),
    ]

    summary = {
        "world_size": world_size,
        "numel": args.numel,
        "padded_numel": padded_numel,
        "shard_numel": shard_numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "transport": args.transport,
        "topology_method": args.topology_method if args.transport == "topology" else None,
        "measurement_order": "full-shard-shard-full",
        "compressed_full_restore_ms": full_restore_ms,
        "ccdl_shard_ms": ccdl_ms,
        "speedup_over_full_restore": full_restore_ms / ccdl_ms,
        "latency_ratio_shard_over_full_restore": ccdl_ms / full_restore_ms,
        "shard_beats_full_restore": ccdl_ms <= full_restore_ms,
        "relative_l2": error,
        "rank_metadata": rank_metadata,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
        "benchmark_identity": identity,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if args.transport == "compressed" and (ccdl_ms > full_restore_ms or ccdl_peak > full_restore_peak):
        raise AssertionError(
            "ReducedShard performance gate failed: shard output must beat full restore "
            "in latency and peak allocated memory"
        )


if __name__ == "__main__":
    run()
