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
_REQUIRED_RESULT_FIELDS = frozenset(
    {
        "world_size",
        "bucket_mib",
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
    if _as_int_or_none(result["fused_kernel_launches"]) != 1:
        failures.append(f"{label}: expected one fused dequant-reduce-mean launch")
    profiler = result["profiler"]
    if (
        not isinstance(profiler, Mapping)
        or _as_int_or_none(profiler.get("production_fused_kernel_launches")) != 1
    ):
        failures.append(
            f"{label}: profiler did not confirm one production fused kernel launch"
        )
    if isinstance(profiler, Mapping) and _as_int_or_none(
        profiler.get("fallback_kernel_launches")
    ) not in {0}:
        failures.append(f"{label}: profiler observed fallback kernel launches")
    if _as_int_or_none(result["steady_allocation_bytes"]) != 0:
        failures.append(f"{label}: steady-state allocation is non-zero")
    allocation = result["allocation_evidence"]
    if not isinstance(allocation, Mapping) or allocation.get(
        "allocated_before_bytes"
    ) != allocation.get("allocated_after_bytes"):
        failures.append(f"{label}: allocation evidence does not show a steady state")
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
    if (
        isinstance(metadata, Mapping)
        and metadata.get("output_ownership") != result["output_mode"]
    ):
        failures.append(f"{label}: fused metadata output ownership does not match mode")
    if not bool(result["no_full_gradient_restoration"]):
        failures.append(f"{label}: candidate path restored a full gradient")
    relative_l2 = _as_float_or_none(result["relative_l2"])
    non_finite = _as_int_or_none(result["non_finite"])
    if relative_l2 is None or relative_l2 > _MAX_RELATIVE_L2 or non_finite != 0:
        failures.append(f"{label}: accuracy gate failed")
    return failures


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
