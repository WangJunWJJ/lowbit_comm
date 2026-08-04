"""Evaluate the Task 12.1 fused ReducedShard A6000 performance gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_WORLD_SIZES = (2, 4)
_BUCKET_MIB = (1, 16, 64)
_OUTPUT_MODES = ("caller", "lease")
_LARGE_BUCKET_MIB = frozenset((16, 64))
_RUNS_PER_CASE = 5
_MAX_RELATIVE_L2 = 0.02
_MEASUREMENT_ORDER = "task12-fused-fused-task12"
_ABBA_POSITIONS = (
    "task12_first",
    "fused_first",
    "fused_second",
    "task12_second",
)
_REQUIRED_RESULT_FIELDS = frozenset(
    {
        "world_size",
        "bucket_mib",
        "numel",
        "dtype",
        "bit",
        "group_size",
        "warmup",
        "repeat",
        "seed",
        "reference",
        "output_mode",
        "measurement_order",
        "task12_ms",
        "fused_ms",
        "speedup",
        "task12_peak_memory_bytes",
        "fused_peak_memory_bytes",
        "steady_allocation_bytes",
        "relative_l2",
        "max_abs_error",
        "non_finite",
        "fused_kernel_launches",
        "fallback_used",
        "output_pointer_stable",
        "output_pointers",
        "per_position_samples_ms",
        "per_position_medians_ms",
        "fused_metadata",
        "profiler",
        "allocation_evidence",
        "identity",
        "no_full_gradient_restoration",
        "run_id",
        "started_at",
        "rank_evidence",
    }
)


def evaluate(results: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return all Task 12.1 gate failures without hiding incomplete evidence.

    Args:
        results: Rank-zero JSON records produced by the fused ReducedShard
            benchmark.

    Returns:
        Human-readable gate failures. An empty list means the submitted
        evidence covers the complete 2/4-GPU matrix and passes every check.
    """

    failures: list[str] = []
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, result in enumerate(results):
        label = f"result[{index}]"
        missing = sorted(_REQUIRED_RESULT_FIELDS.difference(result))
        if missing:
            failures.append(f"{label}: missing fields {missing}")
            continue
        try:
            world_size = _as_int(result["world_size"], "world_size")
            bucket_mib = _as_int(result["bucket_mib"], "bucket_mib")
            output_mode = str(result["output_mode"])
        except ValueError as exc:
            failures.append(f"{label}: {exc}")
            continue
        case_label = _case_label(world_size, bucket_mib, output_mode)
        if (
            world_size not in _WORLD_SIZES
            or bucket_mib not in _BUCKET_MIB
            or output_mode not in _OUTPUT_MODES
        ):
            failures.append(f"{case_label}: unsupported required-matrix case")
            continue
        grouped[(world_size, bucket_mib, output_mode)].append(result)
        failures.extend(_validate_record(result, case_label))

    for world_size in _WORLD_SIZES:
        for bucket_mib in _BUCKET_MIB:
            caller = grouped.get((world_size, bucket_mib, "caller"), [])
            lease = grouped.get((world_size, bucket_mib, "lease"), [])
            for mode, case in (("caller", caller), ("lease", lease)):
                case_label = _case_label(world_size, bucket_mib, mode)
                if len(case) != _RUNS_PER_CASE:
                    failures.append(
                        f"{case_label}: requires exactly {_RUNS_PER_CASE} runs; received {len(case)}"
                    )
                run_ids = {str(result.get("run_id", "")) for result in case}
                if len(run_ids) != _RUNS_PER_CASE:
                    failures.append(f"{case_label}: requires 5 unique run_id values")
                if bucket_mib in _LARGE_BUCKET_MIB and case:
                    task12_ms = _median(case, "task12_ms", case_label, failures)
                    fused_ms = _median(case, "fused_ms", case_label, failures)
                    if (
                        task12_ms is not None
                        and fused_ms is not None
                        and fused_ms > task12_ms
                    ):
                        failures.append(
                            f"{case_label}: large-bucket latency regressed "
                            f"{fused_ms:.6f} > {task12_ms:.6f} ms"
                        )
            if caller and lease:
                caller_l2 = _median(
                    caller,
                    "relative_l2",
                    _case_label(world_size, bucket_mib, "caller"),
                    failures,
                )
                lease_l2 = _median(
                    lease,
                    "relative_l2",
                    _case_label(world_size, bucket_mib, "lease"),
                    failures,
                )
                if (
                    caller_l2 is not None
                    and lease_l2 is not None
                    and not math.isclose(
                        caller_l2, lease_l2, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    failures.append(
                        f"{world_size}gpu/{bucket_mib}MiB: caller/lease accuracy mismatch "
                        f"{caller_l2:.12g} != {lease_l2:.12g}"
                    )
    return failures


def _validate_record(result: Mapping[str, Any], label: str) -> list[str]:
    """Validate the production-path evidence recorded by one independent run."""

    failures: list[str] = []
    if result["measurement_order"] != _MEASUREMENT_ORDER:
        failures.append(
            f"{label}: unbalanced measurement order {result['measurement_order']!r}"
        )
    if bool(result["fallback_used"]):
        failures.append(f"{label}: production fused path used fallback")
    fused_kernel_launches = _as_int_or_none(result["fused_kernel_launches"])
    if fused_kernel_launches != 1:
        failures.append(f"{label}: expected one fused dequant-reduce-mean launch")
    failures.extend(
        _profiler_failures(result["profiler"], fused_kernel_launches, label)
    )
    failures.extend(_timing_evidence_failures(result, label))
    if _as_int_or_none(result["steady_allocation_bytes"]) != 0:
        failures.append(f"{label}: steady-state allocation is non-zero")
    failures.extend(_allocation_failures(result["allocation_evidence"], label))
    if not bool(result["output_pointer_stable"]):
        failures.append(f"{label}: output pointer is unstable")
    pointers = result["output_pointers"]
    if not isinstance(pointers, list) or not pointers or len(set(pointers)) != 1:
        failures.append(f"{label}: output pointer samples are not stable")
    metadata = result["fused_metadata"]
    if not isinstance(metadata, Mapping) or not bool(
        metadata.get("fused_dequant_reduce")
    ):
        failures.append(f"{label}: fused metadata does not confirm production fusion")
    if isinstance(metadata, Mapping) and metadata.get("output_ownership") != "caller":
        failures.append(
            f"{label}: transport metadata must report caller-owned raw output"
        )
    relative_l2 = _as_float_or_none(result["relative_l2"])
    non_finite = _as_int_or_none(result["non_finite"])
    if relative_l2 is None or relative_l2 > _MAX_RELATIVE_L2 or non_finite != 0:
        failures.append(f"{label}: accuracy gate failed")
    expected_values = {
        "dtype": "fp16",
        "bit": 8,
        "group_size": 64,
        "warmup": 20,
        "repeat": 100,
        "seed": 20260802,
        "reference": "fp16_all_reduce",
    }
    for field, expected in expected_values.items():
        if result[field] != expected:
            rendered = repr(expected) if isinstance(expected, str) else str(expected)
            failures.append(f"{label}: requires {field}={rendered}")
    rank_evidence = result["rank_evidence"]
    if not isinstance(rank_evidence, list):
        failures.append(f"{label}: rank evidence must be a list")
    elif len(rank_evidence) != result["world_size"]:
        failures.append(f"{label}: rank evidence count does not match world size")
    else:
        ranks = [
            _as_int_or_none(evidence.get("rank"))
            if isinstance(evidence, Mapping)
            else None
            for evidence in rank_evidence
        ]
        if (
            sorted(rank for rank in ranks if rank is not None)
            != list(range(result["world_size"]))
            or len(set(ranks)) != result["world_size"]
        ):
            failures.append(
                f"{label}: rank evidence must contain each rank exactly once"
            )
        shard_proofs: list[bool] = []
        for evidence in rank_evidence:
            if not isinstance(evidence, Mapping):
                failures.append(f"{label}: rank evidence must contain objects")
                shard_proofs.append(False)
                continue
            proof = _proves_sharded_output_evidence(
                evidence.get("shard_evidence"),
                expected_original_numel=result["numel"],
                expected_world_size=result["world_size"],
            )
            shard_proofs.append(proof)
            failures.extend(
                _rank_evidence_failures(
                    evidence,
                    label,
                    expected_original_numel=result["numel"],
                    expected_world_size=result["world_size"],
                )
            )
        if bool(result["no_full_gradient_restoration"]) != all(shard_proofs):
            failures.append(f"{label}: top-level sharded output proof is inconsistent")
    return failures


def _rank_evidence_failures(
    evidence: Mapping[str, Any],
    label: str,
    *,
    expected_original_numel: int,
    expected_world_size: int,
) -> list[str]:
    """Validate the worst-case evidence collected from every participating rank."""

    rank = _as_int_or_none(evidence.get("rank"))
    rank_label = f"{label}: rank {rank if rank is not None else '?'}"
    failures: list[str] = []
    sharded_output = _proves_sharded_output_evidence(
        evidence.get("shard_evidence"),
        expected_original_numel=expected_original_numel,
        expected_world_size=expected_world_size,
    )
    if (
        not sharded_output
        or evidence.get("no_full_gradient_restoration") is not sharded_output
    ):
        failures.append(f"{rank_label}: sharded output proof failed")
    if bool(evidence.get("fallback_used")):
        failures.append(f"{rank_label}: production fused path used fallback")
    fused_kernel_launches = _as_int_or_none(evidence.get("fused_kernel_launches"))
    if fused_kernel_launches != 1:
        failures.append(f"{rank_label}: expected one fused dequant-reduce-mean launch")
    failures.extend(
        _profiler_failures(evidence.get("profiler"), fused_kernel_launches, rank_label)
    )
    if not bool(evidence.get("output_pointer_stable")):
        failures.append(f"{rank_label}: output pointer is unstable")
    pointers = evidence.get("output_pointers")
    if not isinstance(pointers, list) or not pointers or len(set(pointers)) != 1:
        failures.append(f"{rank_label}: output pointer samples are not stable")
    metadata = evidence.get("fused_metadata")
    if not isinstance(metadata, Mapping) or not bool(
        metadata.get("fused_dequant_reduce")
    ):
        failures.append(
            f"{rank_label}: fused metadata does not confirm production fusion"
        )
    if isinstance(metadata, Mapping) and metadata.get("output_ownership") != "caller":
        failures.append(
            f"{rank_label}: transport metadata must report caller-owned raw output"
        )
    relative_l2 = _as_float_or_none(evidence.get("relative_l2"))
    non_finite = _as_int_or_none(evidence.get("non_finite"))
    if relative_l2 is None or relative_l2 > _MAX_RELATIVE_L2 or non_finite != 0:
        failures.append(f"{rank_label}: accuracy gate failed")
    failures.extend(
        _allocation_failures(evidence.get("allocation_evidence"), rank_label)
    )
    return failures


def _proves_sharded_output_evidence(
    evidence: object,
    *,
    expected_original_numel: object,
    expected_world_size: object,
) -> bool:
    """Recompute sharded-output truth from rank-local runtime facts."""

    if not isinstance(evidence, Mapping):
        return False
    required = (
        "tensor_numel",
        "shard_numel",
        "padded_numel",
        "original_numel",
        "world_size",
        "transport",
    )
    if any(field not in evidence for field in required):
        return False
    values = {
        field: _as_int_or_none(evidence.get(field))
        for field in required
        if field != "transport"
    }
    if any(value is None for value in values.values()):
        return False
    tensor_numel = int(values["tensor_numel"])
    shard_numel = int(values["shard_numel"])
    padded_numel = int(values["padded_numel"])
    original_numel = int(values["original_numel"])
    world_size = int(values["world_size"])
    return (
        _as_int_or_none(expected_world_size) is not None
        and _as_int_or_none(expected_original_numel) is not None
        and int(expected_world_size) > 1
        and world_size == int(expected_world_size)
        and tensor_numel == shard_numel
        and 0 < shard_numel < padded_numel
        and padded_numel == shard_numel * world_size
        and original_numel == int(expected_original_numel)
        and original_numel <= padded_numel
        and evidence["transport"] == "compressed_all_to_all"
    )


def _profiler_failures(
    profiler: object,
    recorded_launches: int | None,
    label: str,
) -> list[str]:
    """Require profiler names and aggregate launch count to prove fusion."""

    if not isinstance(profiler, Mapping):
        return [f"{label}: profiler did not record all observed kernel names"]
    kernel_names = profiler.get("kernel_names")
    fused_names = profiler.get("production_fused_kernel_names")
    profiler_launches = _as_int_or_none(
        profiler.get("production_fused_kernel_launches")
    )
    if not isinstance(kernel_names, list) or not all(
        isinstance(name, str) for name in kernel_names
    ):
        return [f"{label}: profiler did not record all observed kernel names"]
    observed_fused = [name for name in kernel_names if "dequant_reduce_fused_" in name]
    failures: list[str] = []
    if (
        not isinstance(fused_names, list)
        or not fused_names
        or not all(
            isinstance(name, str) and "dequant_reduce_fused_" in name
            for name in fused_names
        )
        or fused_names != observed_fused
    ):
        failures.append(f"{label}: profiler kernel names do not prove fused execution")
    if profiler_launches != 1 or profiler_launches != recorded_launches:
        failures.append(f"{label}: profiler launch count is inconsistent")
    return failures


def _timing_evidence_failures(
    result: Mapping[str, Any],
    label: str,
) -> list[str]:
    """Validate measured ABBA positions rather than trusting the order label."""

    samples = result["per_position_samples_ms"]
    medians = result["per_position_medians_ms"]
    if not isinstance(samples, Mapping) or not isinstance(medians, Mapping):
        return [f"{label}: ABBA evidence must contain exactly {_ABBA_POSITIONS}"]
    failures: list[str] = []
    if set(samples) != set(_ABBA_POSITIONS) or set(medians) != set(_ABBA_POSITIONS):
        failures.append(
            f"{label}: ABBA evidence must contain exactly {_ABBA_POSITIONS}"
        )
    valid_position_medians: dict[str, float] = {}
    for position in _ABBA_POSITIONS:
        position_samples = samples.get(position)
        supplied_median = _as_float_or_none(medians.get(position))
        if not isinstance(position_samples, list) or not position_samples:
            failures.append(f"{label}: timing samples must be finite and non-negative")
            continue
        numeric_samples = [_as_float_or_none(value) for value in position_samples]
        if any(value is None for value in numeric_samples):
            failures.append(f"{label}: timing samples must be finite and non-negative")
            continue
        actual_median = statistics.median(
            value for value in numeric_samples if value is not None
        )
        if supplied_median is None or not math.isclose(
            supplied_median, actual_median, rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"{label}: position median does not match samples")
            continue
        valid_position_medians[position] = supplied_median
    if len(valid_position_medians) == len(_ABBA_POSITIONS):
        task12_median = statistics.median(
            (
                valid_position_medians["task12_first"],
                valid_position_medians["task12_second"],
            )
        )
        fused_median = statistics.median(
            (
                valid_position_medians["fused_first"],
                valid_position_medians["fused_second"],
            )
        )
        if not math.isclose(
            float(result["task12_ms"]), task12_median, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            float(result["fused_ms"]), fused_median, rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"{label}: aggregate median does not match ABBA evidence")
    return failures


def _allocation_failures(allocation: object, label: str) -> list[str]:
    """Require both zero peak growth and a restored current allocation watermark."""

    if not isinstance(allocation, Mapping):
        return [f"{label}: allocation evidence does not show a steady state"]
    before = _as_int_or_none(allocation.get("allocated_before_bytes"))
    peak = _as_int_or_none(allocation.get("candidate_peak_bytes"))
    after = _as_int_or_none(allocation.get("allocated_after_bytes"))
    if before is None or peak is None or after is None or after != before:
        return [f"{label}: allocation evidence does not show a steady state"]
    if max(0, peak - before) != 0:
        return [f"{label}: steady-state allocation is non-zero"]
    return []


def _median(
    results: list[Mapping[str, Any]], field: str, label: str, failures: list[str]
) -> float | None:
    values = [_as_float_or_none(result.get(field)) for result in results]
    if any(value is None for value in values):
        failures.append(f"{label}: {field} must be a finite non-negative number")
        return None
    return statistics.median(value for value in values if value is not None)


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _as_int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def _case_label(world_size: int, bucket_mib: int, output_mode: str) -> str:
    return f"{world_size}gpu/{bucket_mib}MiB/{output_mode}"


def main() -> int:
    """Load benchmark JSON files, print every failure, and return gate status."""

    parser = argparse.ArgumentParser(
        description="Gate fused ReducedShard performance evidence"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    failures = evaluate(results)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Task 12.1 fused ReducedShard gate passed for {len(results)} result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
