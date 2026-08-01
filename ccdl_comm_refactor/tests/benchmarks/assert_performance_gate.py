"""Compare like-for-like CCDL benchmark results against a frozen baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    from tests.benchmarks.result_schema import validate_result
except ModuleNotFoundError as exc:
    if exc.name != "tests":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.benchmarks.result_schema import validate_result


COMPARABLE_FIELDS = ("gpu_name", "world_size", "dtype", "numel", "strategy")


def compare_results(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    max_regression: float,
) -> list[str]:
    """Return deterministic diagnostics for incomparable or regressed results."""

    validate_result(baseline)
    validate_result(candidate)
    if max_regression < 0.0:
        raise ValueError("max_regression must be non-negative")

    failures = []
    for field in COMPARABLE_FIELDS:
        if baseline[field] != candidate[field]:
            failures.append(
                f"incomparable field {field}: "
                f"baseline={baseline[field]}, candidate={candidate[field]}"
            )
    if failures:
        return failures

    ratio = float(candidate["latency_ms"]) / float(baseline["latency_ms"])
    if ratio > 1.0 + max_regression:
        failures.append(f"latency regression: {ratio:.4f}")
    return failures


def check_metric_maximum(
    candidate: dict[str, object],
    *,
    metric: str,
    maximum: float,
) -> list[str]:
    """Return a diagnostic when one standalone metric exceeds its maximum."""

    if metric not in candidate:
        raise ValueError(f"missing metric: {metric}")
    value = candidate[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {metric} must be finite numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not math.isfinite(maximum):
        raise ValueError(f"metric {metric} and maximum must be finite numeric")
    if numeric > maximum:
        return [f"metric {metric} exceeds maximum: {numeric:.6f} > {maximum:.6f}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, nargs="?")
    parser.add_argument("positional_candidate", type=Path, nargs="?")
    parser.add_argument("--candidate", dest="candidate_option", type=Path)
    parser.add_argument("--metric")
    parser.add_argument("--max", dest="maximum", type=float)
    parser.add_argument("--max-regression", type=float, default=0.02)
    args = parser.parse_args()
    if args.candidate_option is not None or args.metric is not None or args.maximum is not None:
        if args.candidate_option is None or args.metric is None or args.maximum is None:
            parser.error("--candidate, --metric, and --max must be provided together")
        candidate = json.loads(args.candidate_option.read_text(encoding="utf-8"))
        failures = check_metric_maximum(
            candidate,
            metric=args.metric,
            maximum=args.maximum,
        )
    else:
        if args.baseline is None or args.positional_candidate is None:
            parser.error("baseline and candidate are required for comparison mode")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.positional_candidate.read_text(encoding="utf-8"))
        failures = compare_results(baseline, candidate, max_regression=args.max_regression)
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
