"""Benchmark Task 12 versus Task 12.1 fused ReducedShard output paths.

Run under ``torchrun``.  The script deliberately writes one rank-zero JSON
record per independent invocation so the gate can require five fresh runs for
every 2/4-GPU, bucket-size, and output-ownership combination.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from ccdl_comm import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus, load_cuda_extension
from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
from tests.benchmarks.result_schema import resolve_benchmark_identity


_ABBA_POSITIONS = (
    ("task12_first", "task12"),
    ("fused_first", "fused"),
    ("fused_second", "fused"),
    ("task12_second", "task12"),
)
_KERNEL_MARKER = "dequant_reduce_fused_"
torch: Any = None
dist: Any = None
ProfilerActivity: Any = None
profile: Any = None


def _abba_positions() -> tuple[tuple[str, str], ...]:
    """Return deterministic output keys and operation names for ABBA timing."""

    return _ABBA_POSITIONS


def _reduced_shard_evidence(reduced: object) -> dict[str, object]:
    """Extract shape and transport facts from one runtime ReducedShard."""

    shard = getattr(reduced, "shard")
    return {
        "tensor_numel": int(shard.numel()),
        "shard_numel": int(getattr(reduced, "shard_numel")),
        "padded_numel": int(getattr(reduced, "padded_numel")),
        "original_numel": int(getattr(reduced, "original_numel")),
        "world_size": int(getattr(reduced, "world_size")),
        "transport": str(getattr(reduced, "transport")),
    }


def _proves_sharded_output(
    evidence: object,
    *,
    expected_original_numel: int,
    expected_world_size: int,
) -> bool:
    """Derive whether runtime facts prove a rank-local compressed shard."""

    if not isinstance(evidence, dict):
        return False
    required = {
        "tensor_numel",
        "shard_numel",
        "padded_numel",
        "original_numel",
        "world_size",
        "transport",
    }
    if set(evidence) != required:
        return False
    integer_fields = (
        "tensor_numel",
        "shard_numel",
        "padded_numel",
        "original_numel",
        "world_size",
    )
    if any(
        isinstance(evidence[field], bool) or not isinstance(evidence[field], int)
        for field in integer_fields
    ):
        return False
    tensor_numel = int(evidence["tensor_numel"])
    shard_numel = int(evidence["shard_numel"])
    padded_numel = int(evidence["padded_numel"])
    original_numel = int(evidence["original_numel"])
    world_size = int(evidence["world_size"])
    return (
        expected_world_size > 1
        and world_size == expected_world_size
        and tensor_numel == shard_numel
        and 0 < shard_numel < padded_numel
        and padded_numel == shard_numel * world_size
        and original_numel == expected_original_numel
        and original_numel <= padded_numel
        and evidence["transport"] == "compressed_all_to_all"
    )


def parse_args() -> argparse.Namespace:
    """Parse the fixed Task 12.1 benchmark matrix parameters."""

    parser = argparse.ArgumentParser(
        description="Benchmark fused ReducedShard output paths"
    )
    parser.add_argument("--bucket-mib", type=int, choices=(1, 16, 64), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", choices=("caller", "lease"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_torch_runtime() -> None:
    """Load PyTorch only for distributed execution, keeping contract tests CPU-only."""

    global ProfilerActivity, dist, profile, torch
    if torch is not None:
        return
    torch = import_module("torch")
    dist = import_module("torch.distributed")
    profiler = import_module("torch.profiler")
    ProfilerActivity = profiler.ProfilerActivity
    profile = profiler.profile


class _FusedMeanDisabledExtension:
    """Delegate native extension calls while making only fused mean unavailable."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> object:
        if name == "inplace_dequantize_reduce_mean":
            raise AttributeError(name)
        return getattr(self._delegate, name)


def _baseline_extension_status(status: CudaExtensionStatus) -> CudaExtensionStatus:
    """Return a usable extension status whose Task 12 callback is disabled."""

    if not status.available or status.module is None:
        raise RuntimeError(status.reason or "CCDL CUDA extension is unavailable")
    return CudaExtensionStatus(
        available=True,
        module=_FusedMeanDisabledExtension(status.module),
        reason=status.reason,
    )


def _setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def _max_across_ranks(value: float | int, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _measure_position(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[float, int]:
    """Measure one ABBA position without timing synchronization or barriers."""

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    for _ in range(repeat):
        operation()
    torch.cuda.synchronize(device)
    elapsed_ms = (perf_counter() - start) * 1000.0 / repeat
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    return _max_across_ranks(elapsed_ms, device=device), int(
        _max_across_ranks(peak_memory, device=device)
    )


def _reference_shard(
    source: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Compute one FP16/BF16/FP32 reference outside every timed operation."""

    shard_numel = (source.numel() + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    if padded_numel == source.numel():
        padded = source.clone()
    else:
        padded = torch.cat(
            (source, source.new_zeros((padded_numel - source.numel(),))), dim=0
        )
    dist.all_reduce(padded, op=dist.ReduceOp.SUM)
    padded.div_(world_size)
    return padded.narrow(0, rank * shard_numel, shard_numel).contiguous()


def _error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | int]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    return {
        "relative_l2": float(
            (difference.norm() / reference_f32.norm().clamp_min(1e-12)).item()
        ),
        "max_abs_error": float(difference.abs().max().item()),
        "non_finite": int((~torch.isfinite(candidate_f32)).sum().item()),
    }


def _profile_fused_kernel(
    operation: Callable[[], Any], *, device: torch.device
) -> dict[str, object]:
    """Capture one candidate call outside timing and count its production kernel."""

    with profile(activities=[ProfilerActivity.CUDA]) as active_profile:
        operation()
        torch.cuda.synchronize(device)
    return _profiler_evidence(list(active_profile.key_averages()))


def _profiler_evidence(events: list[Any]) -> dict[str, object]:
    """Return complete profiler evidence without inventing fallback counters."""

    kernel_names = [str(getattr(event, "key", "")) for event in events]
    fused_names = [name for name in kernel_names if _KERNEL_MARKER in name]
    return {
        "kernel_names": kernel_names,
        "production_fused_kernel_names": fused_names,
        "production_fused_kernel_launches": _fused_kernel_launch_count(events),
    }


def _fused_kernel_launch_count(events: list[Any]) -> int:
    """Count launches from profiler aggregates instead of distinct kernel names."""

    return sum(
        int(getattr(event, "count", 0))
        for event in events
        if _KERNEL_MARKER in str(getattr(event, "key", ""))
    )


def _require_fused_metadata(metadata: dict[str, object]) -> None:
    if not metadata.get("fused_dequant_reduce"):
        reason = metadata.get("fused_dequant_reduce_reason", "unknown reason")
        raise RuntimeError(f"fused ReducedShard callback declined: {reason}")


def run() -> None:
    """Execute one independent Task 12.1 benchmark record and write rank-zero JSON."""

    _load_torch_runtime()
    args = parse_args()
    rank, world_size, device = _setup()
    try:
        extension_status = load_cuda_extension()
        if not extension_status.available or extension_status.module is None:
            raise RuntimeError(
                extension_status.reason or "CCDL CUDA extension is unavailable"
            )
        if (
            args.dtype != "fp16"
            or args.bit != 8
            or args.group_size != 64
            or args.warmup != 20
            or args.repeat != 100
            or args.seed != 20260802
        ):
            raise ValueError(
                "Task 12.1 fused gate requires fp16, INT8/group64, warmup=20, "
                "repeat=100, and seed=20260802"
            )
        dtype = _dtype(args.dtype)
        numel = (
            args.bucket_mib * 1024 * 1024 // torch.empty((), dtype=dtype).element_size()
        )
        config = CompressionConfig(bit=args.bit, group_size=args.group_size)
        torch.manual_seed(args.seed + rank)
        source = torch.randn(numel, dtype=dtype, device=device) * 0.1 + rank * 0.01
        reference = _reference_shard(source, rank=rank, world_size=world_size)
        chunk_plan = compile_chunk_plan(original_numel=numel, world_size=world_size)

        task12_plan = compile_cuda_shortcut(
            source,
            collective="reduce_scatter",
            strategy="compressed",
            output_layout="shard",
            config=config,
            async_op=False,
            dtype=args.dtype,
            extension_status=_baseline_extension_status(extension_status),
        )
        fused_plan = compile_cuda_shortcut(
            source,
            collective="reduce_scatter",
            strategy="compressed",
            output_layout="shard",
            config=config,
            async_op=False,
            dtype=args.dtype,
            extension_status=extension_status,
        )
        caller_output = (
            source.new_empty((chunk_plan.shard_numel,))
            if args.mode == "caller"
            else None
        )
        output_pointers: list[int] = []
        final_metadata: dict[str, object] = {}
        final_candidate: torch.Tensor | None = None
        final_shard_evidence: dict[str, object] = {}

        def task12_once() -> torch.Tensor:
            return task12_plan.run(source).wait().shard

        def fused_once() -> torch.Tensor:
            nonlocal final_candidate, final_metadata, final_shard_evidence
            lease = (
                fused_plan.executor.acquire_output() if args.mode == "lease" else None
            )
            output = lease if lease is not None else caller_output
            if output is None:
                raise AssertionError("caller mode requires a stable output")
            reduced = fused_plan.run(source, out=output).wait()
            shard = reduced.shard
            _require_fused_metadata(dict(reduced.metadata))
            output_pointers.append(int(shard.data_ptr()))
            final_candidate = shard
            final_metadata = dict(reduced.metadata)
            final_shard_evidence = _reduced_shard_evidence(reduced)
            if not _proves_sharded_output(
                final_shard_evidence,
                expected_original_numel=numel,
                expected_world_size=world_size,
            ):
                raise AssertionError(
                    f"ReducedShard output proof failed: {final_shard_evidence}"
                )
            if lease is not None:
                lease.release_after(shard)
            return shard

        # Warm the pool/allocator once before the independent steady-allocation check.
        fused_once()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        fused_once()
        torch.cuda.synchronize(device)
        candidate_peak_bytes = int(torch.cuda.max_memory_allocated(device))
        allocated_after = int(torch.cuda.memory_allocated(device))
        steady_allocation_bytes = max(0, candidate_peak_bytes - allocated_before)

        position_samples: dict[str, list[float]] = {}
        position_peaks: dict[str, int] = {}
        for key, operation_name in _abba_positions():
            operation = task12_once if operation_name == "task12" else fused_once
            latency, peak = _measure_position(
                operation,
                warmup=args.warmup,
                repeat=args.repeat,
                device=device,
            )
            position_samples[key] = [latency]
            position_peaks[key] = peak

        # This separate call is intentionally outside timing and proves metadata/accuracy.
        candidate = fused_once() if final_candidate is None else final_candidate
        torch.cuda.synchronize(device)
        metrics = _error_metrics(reference, candidate)
        if metrics["relative_l2"] > 0.02 or metrics["non_finite"] != 0:
            raise AssertionError(f"fused ReducedShard accuracy gate failed: {metrics}")
        profiler = _profile_fused_kernel(fused_once, device=device)
        if profiler["production_fused_kernel_launches"] != 1:
            raise AssertionError(f"expected one fused kernel launch, got {profiler}")

        task12_ms = statistics.median(
            (position_samples["task12_first"][0], position_samples["task12_second"][0])
        )
        fused_ms = statistics.median(
            (position_samples["fused_first"][0], position_samples["fused_second"][0])
        )
        pointer_stable = bool(output_pointers) and len(set(output_pointers)) == 1
        local_no_full_gradient_restoration = _proves_sharded_output(
            final_shard_evidence,
            expected_original_numel=numel,
            expected_world_size=world_size,
        )
        rank_evidence: list[object | None] = [None] * world_size
        local_rank_evidence = {
            "rank": rank,
            "shard_evidence": final_shard_evidence,
            "no_full_gradient_restoration": local_no_full_gradient_restoration,
            "relative_l2": metrics["relative_l2"],
            "max_abs_error": metrics["max_abs_error"],
            "non_finite": metrics["non_finite"],
            "fused_kernel_launches": profiler["production_fused_kernel_launches"],
            "fallback_used": not bool(final_metadata.get("fused_dequant_reduce")),
            "fallback_reason": final_metadata.get("fused_dequant_reduce_reason"),
            "output_pointer_stable": pointer_stable,
            "output_pointers": output_pointers,
            "fused_metadata": final_metadata,
            "profiler": profiler,
            "steady_allocation_bytes": steady_allocation_bytes,
            "allocation_evidence": {
                "allocated_before_bytes": allocated_before,
                "candidate_peak_bytes": candidate_peak_bytes,
                "allocated_after_bytes": allocated_after,
            },
        }
        dist.all_gather_object(rank_evidence, local_rank_evidence)
        completed_rank_evidence = [
            item for item in rank_evidence if isinstance(item, dict)
        ]
        if len(completed_rank_evidence) != world_size:
            raise RuntimeError("failed to gather complete per-rank benchmark evidence")
        worst_allocation = max(
            completed_rank_evidence,
            key=lambda item: max(
                0,
                int(item["allocation_evidence"]["candidate_peak_bytes"])
                - int(item["allocation_evidence"]["allocated_before_bytes"]),
            ),
        )
        worst_relative_l2 = max(
            float(item["relative_l2"]) for item in completed_rank_evidence
        )
        worst_max_abs_error = max(
            float(item["max_abs_error"]) for item in completed_rank_evidence
        )
        worst_non_finite = max(
            int(item["non_finite"]) for item in completed_rank_evidence
        )
        worst_fused_launches = max(
            int(item["fused_kernel_launches"]) for item in completed_rank_evidence
        )
        any_fallback = any(
            bool(item["fallback_used"]) for item in completed_rank_evidence
        )
        all_output_pointers_stable = all(
            bool(item["output_pointer_stable"]) for item in completed_rank_evidence
        )
        worst_steady_allocation = max(
            int(item["steady_allocation_bytes"]) for item in completed_rank_evidence
        )
        all_sharded_outputs = all(
            _proves_sharded_output(
                item["shard_evidence"],
                expected_original_numel=numel,
                expected_world_size=world_size,
            )
            for item in completed_rank_evidence
        )
        run_envelope: list[object | None] = [
            {
                "run_id": uuid4().hex,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(run_envelope, src=0)
        if not isinstance(run_envelope[0], dict):
            raise RuntimeError("rank zero did not broadcast benchmark run identity")
        run_identity = run_envelope[0]
        result = {
            "benchmark": "task12_1_fused_reduced_shard",
            "identity": resolve_benchmark_identity(),
            "world_size": world_size,
            "bucket_mib": args.bucket_mib,
            "numel": numel,
            "dtype": args.dtype,
            "bit": args.bit,
            "group_size": args.group_size,
            "output_mode": args.mode,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "measurement_order": "task12-fused-fused-task12",
            "task12_ms": task12_ms,
            "fused_ms": fused_ms,
            "speedup": task12_ms / fused_ms,
            "task12_peak_memory_bytes": max(
                position_peaks["task12_first"], position_peaks["task12_second"]
            ),
            "fused_peak_memory_bytes": max(
                position_peaks["fused_first"], position_peaks["fused_second"]
            ),
            "steady_allocation_bytes": worst_steady_allocation,
            "output_pointer_stable": all_output_pointers_stable,
            "output_pointers": output_pointers,
            "per_position_samples_ms": position_samples,
            "per_position_medians_ms": {
                key: values[0] for key, values in position_samples.items()
            },
            "fused_metadata": final_metadata,
            "rank_evidence": completed_rank_evidence,
            "profiler": profiler,
            "fused_kernel_launches": worst_fused_launches,
            "fallback_used": any_fallback,
            "fallback_reason": final_metadata.get("fused_dequant_reduce_reason"),
            "allocation_evidence": {
                **worst_allocation["allocation_evidence"],
            },
            "no_full_gradient_restoration": all_sharded_outputs,
            "seed": args.seed,
            "reference": "fp16_all_reduce",
            "run_id": run_identity["run_id"],
            "started_at": run_identity["started_at"],
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "relative_l2": worst_relative_l2,
            "max_abs_error": worst_max_abs_error,
            "non_finite": worst_non_finite,
        }
        if rank == 0:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
