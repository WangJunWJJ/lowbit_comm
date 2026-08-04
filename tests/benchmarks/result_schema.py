"""Versioned schema checks for reproducible GPU benchmark results."""

from __future__ import annotations

import math
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path


REQUIRED_FIELDS = frozenset(
    {
        "commit",
        "hostname",
        "gpu_name",
        "cuda_version",
        "torch_version",
        "world_size",
        "dtype",
        "numel",
        "strategy",
        "latency_ms",
        "effective_gbps",
        "peak_memory_bytes",
        "relative_l2",
        "max_abs_error",
        "rmse",
        "non_finite",
    }
)

_NON_EMPTY_STRINGS = (
    "commit",
    "hostname",
    "gpu_name",
    "cuda_version",
    "torch_version",
    "dtype",
    "strategy",
)
_POSITIVE_INTEGERS = ("world_size", "numel")
_NON_NEGATIVE_INTEGERS = ("peak_memory_bytes", "non_finite")
_NON_NEGATIVE_FINITE = ("effective_gbps", "relative_l2", "max_abs_error", "rmse")


def _require_integer(payload: Mapping[str, object], field: str, *, positive: bool) -> None:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")


def _require_finite_number(payload: Mapping[str, object], field: str, *, positive: bool) -> None:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < (0.0 if not positive else math.nextafter(0.0, 1.0)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")


def validate_result(payload: Mapping[str, object]) -> None:
    """Validate one benchmark result or raise a diagnostic ``ValueError``."""

    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"missing benchmark fields: {sorted(missing)}")

    for field in _NON_EMPTY_STRINGS:
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    if str(payload["commit"]).strip().lower() in {"unknown", "unset", "none"}:
        raise ValueError("commit must identify the benchmark source revision")
    for field in _POSITIVE_INTEGERS:
        _require_integer(payload, field, positive=True)
    for field in _NON_NEGATIVE_INTEGERS:
        _require_integer(payload, field, positive=False)
    _require_finite_number(payload, "latency_ms", positive=True)
    for field in _NON_NEGATIVE_FINITE:
        _require_finite_number(payload, field, positive=False)


def resolve_benchmark_identity(
    env: Mapping[str, str] = os.environ,
    *,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Resolve stable source and host identity across container boundaries."""

    commit = env.get("CCDL_BENCHMARK_COMMIT", "").strip()
    if not commit:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
    hostname = env.get("CCDL_BENCHMARK_HOSTNAME", "").strip() or platform.node()
    return {"commit": commit or "unknown", "hostname": hostname or "unknown"}
