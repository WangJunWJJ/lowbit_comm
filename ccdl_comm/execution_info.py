"""Immutable diagnostics produced while compiling a communication plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .stage import _require_non_empty


@dataclass(frozen=True)
class ExecutionInfo:
    """Static FR-015 execution metadata exposed by compiled plans and work."""

    requested_strategy: str
    executed_strategy: str
    backend: str
    fallback_used: bool
    fallback_reason: str | None
    stage_names: tuple[str, ...]
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    workspace_cache_hit: bool
    async_capable: bool
    fast_path: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("requested_strategy", "executed_strategy", "backend", "fast_path"):
            _require_non_empty(getattr(self, field_name), field_name)
        stage_names = tuple(self.stage_names)
        for stage_name in stage_names:
            _require_non_empty(stage_name, "stage name")
        object.__setattr__(self, "stage_names", stage_names)

        if self.original_bytes < 0:
            raise ValueError("original_bytes must be >= 0")
        if self.compressed_bytes < 0:
            raise ValueError("compressed_bytes must be >= 0")
        if not math.isfinite(self.compression_ratio) or self.compression_ratio <= 0.0:
            raise ValueError("compression_ratio must be finite and > 0")
        if self.fallback_used:
            if self.fallback_reason is None or not self.fallback_reason.strip():
                raise ValueError("fallback_reason is required when fallback_used is true")
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason must be None when fallback_used is false")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class ExecutionCounterSnapshot:
    """Immutable diagnostic snapshot of executor and work activity."""

    run_calls: int
    completed_runs: int
    failed_runs: int
    wait_calls: int
    query_calls: int


class ExecutionCounters:
    """Preallocated, lock-free counters shared by one executor and its work.

    Snapshots are diagnostic and best-effort when one executor is driven by
    multiple Python threads concurrently. They are not synchronization state.
    """

    __slots__ = (
        "_run_calls",
        "_completed_runs",
        "_failed_runs",
        "_wait_calls",
        "_query_calls",
    )

    def __init__(self) -> None:
        self._run_calls = 0
        self._completed_runs = 0
        self._failed_runs = 0
        self._wait_calls = 0
        self._query_calls = 0

    def snapshot(self) -> ExecutionCounterSnapshot:
        """Create an immutable snapshot on the diagnostics cold path."""

        return ExecutionCounterSnapshot(
            run_calls=self._run_calls,
            completed_runs=self._completed_runs,
            failed_runs=self._failed_runs,
            wait_calls=self._wait_calls,
            query_calls=self._query_calls,
        )

    def _record_run(self) -> None:
        self._run_calls += 1

    def _record_completed(self) -> None:
        self._completed_runs += 1

    def _record_failed(self) -> None:
        self._failed_runs += 1

    def _record_wait(self) -> None:
        self._wait_calls += 1

    def _record_query(self) -> None:
        self._query_calls += 1
