"""Compare like-for-like CCDL benchmark results against a frozen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.02)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    failures = compare_results(baseline, candidate, max_regression=args.max_regression)
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
