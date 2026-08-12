"""Validate Task 11 fused Executor correctness and large-bucket performance."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


_REQUIRED_CASES = {(world_size, bucket_mib) for world_size in (2, 4) for bucket_mib in (1, 16, 64)}
_LARGE_BUCKETS = {16, 64}
_MIN_RUNS = 5
_MAX_TASK0_REGRESSION = 0.02


def measure_balanced(
    measure: Callable[[Callable[[], None]], float],
    baseline: Callable[[], None],
    fused: Callable[[], None],
) -> tuple[float, float]:
    """Measure in ABBA order so neither candidate owns one thermal position."""

    baseline_first = measure(baseline)
    fused_first = measure(fused)
    fused_second = measure(fused)
    baseline_second = measure(baseline)
    return (
        (baseline_first + baseline_second) / 2.0,
        (fused_first + fused_second) / 2.0,
    )


def evaluate(
    results: list[dict[str, Any]],
    *,
    task0_baselines: dict[tuple[int, int], float],
) -> list[str]:
    failures: list[str] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        world_size = int(result["world_size"])
        bucket_mib = int(result["bucket_mib"])
        label = f"{world_size}gpu/{bucket_mib}MiB"
        grouped[(world_size, bucket_mib)].append(result)
        if bool(result["fallback_used"]):
            failures.append(f"{label}: fused Executor used fallback")
        if result.get("error_feedback_reference") != "local_reconstruction":
            failures.append(f"{label}: error feedback is not based on local reconstruction")
        fast_path = str(result.get("fast_path", ""))
        if fast_path != "cuda_fused_dequant_reduce_mean_ef":
            failures.append(f"{label}: unexpected fast path {fast_path!r}")
        measurement_order = str(result.get("measurement_order", ""))
        if measurement_order != "baseline-fused-fused-baseline":
            failures.append(f"{label}: unbalanced measurement order {measurement_order!r}")
        allocation = int(result["steady_allocation_bytes"])
        if allocation != 0:
            failures.append(f"{label}: steady allocation is {allocation} bytes")
        precision_delta = abs(
            float(result["fused_relative_l2"])
            - float(result["baseline_relative_l2"])
        )
        if precision_delta > 1e-9:
            failures.append(f"{label}: precision mismatch is {precision_delta:.12g}")

    for world_size, bucket_mib in sorted(_REQUIRED_CASES):
        group = grouped.get((world_size, bucket_mib), [])
        label = f"{world_size}gpu/{bucket_mib}MiB"
        if not group:
            failures.append(f"missing {label} benchmark case")
            continue
        if len(group) < _MIN_RUNS:
            failures.append(f"{label}: requires {_MIN_RUNS} runs; received {len(group)}")
        if bucket_mib not in _LARGE_BUCKETS:
            continue
        baseline_ms = statistics.median(float(result["baseline_ms"]) for result in group)
        fused_ms = statistics.median(float(result["fused_ms"]) for result in group)
        if fused_ms > baseline_ms:
            failures.append(
                f"{world_size}gpu/{bucket_mib}MiB: latency regression "
                f"{fused_ms:.6f} > {baseline_ms:.6f} ms"
            )
        task0_ms = task0_baselines.get((world_size, bucket_mib))
        if task0_ms is None:
            failures.append(f"{label}: missing Task 0 baseline")
        elif fused_ms > task0_ms * (1.0 + _MAX_TASK0_REGRESSION):
            failures.append(
                f"{label}: Task 0 regression {fused_ms:.6f} > "
                f"{task0_ms * (1.0 + _MAX_TASK0_REGRESSION):.6f} ms"
            )
    return failures


def _load_task0_baselines(root: Path) -> dict[tuple[int, int], float]:
    numel_by_bucket = {1: 524288, 16: 8388608, 64: 33554432}
    baselines = {}
    for world_size, bucket_mib in sorted(_REQUIRED_CASES):
        path = root / f"{world_size}gpu_fp16_{numel_by_bucket[bucket_mib]}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        baselines[(world_size, bucket_mib)] = float(result["ccdl_all_gather_reduce_ms"])
    return baselines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--task0-root",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "gpu_first_baseline" / "raw",
    )
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    failures = evaluate(results, task0_baselines=_load_task0_baselines(args.task0_root))
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Task 11 gate passed for {len(results)} result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
