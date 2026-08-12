"""CUDA timeline primitives for evidence-based overlap reporting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from math import isclose, isfinite
from threading import Lock
from typing import Any, Iterable


class InvalidOverlapMeasurement(ValueError):
    """Raised when timeline fields cannot describe a physical execution."""


@dataclass(frozen=True, slots=True)
class OverlapMeasurement:
    communication_ms: float
    compute_ms: float
    overlapped_ms: float
    exposed_communication_ms: float

    def __post_init__(self) -> None:
        values = (
            self.communication_ms,
            self.compute_ms,
            self.overlapped_ms,
            self.exposed_communication_ms,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise InvalidOverlapMeasurement("timeline durations must be finite and >= 0")
        intersection = self.communication_ms + self.compute_ms - self.overlapped_ms
        tolerance = 1e-6
        if intersection < -tolerance:
            raise InvalidOverlapMeasurement("overlapped_ms exceeds the sum of both intervals")
        if intersection > min(self.communication_ms, self.compute_ms) + tolerance:
            raise InvalidOverlapMeasurement("timeline intersection exceeds a source interval")
        expected_exposed = self.communication_ms - max(0.0, intersection)
        if not isclose(
            self.exposed_communication_ms,
            expected_exposed,
            rel_tol=1e-6,
            abs_tol=tolerance,
        ):
            raise InvalidOverlapMeasurement(
                "exposed communication is inconsistent with the timeline intersection"
            )

    @property
    def intersection_ms(self) -> float:
        return max(
            0.0,
            self.communication_ms + self.compute_ms - self.overlapped_ms,
        )

    def overlap_efficiency(self) -> float:
        denominator = min(self.communication_ms, self.compute_ms)
        if denominator <= 0:
            return 0.0
        efficiency = self.intersection_ms / denominator
        if not 0.0 <= efficiency <= 1.0:
            raise InvalidOverlapMeasurement("derived overlap efficiency is outside [0, 1]")
        return efficiency


def classify_overlap(
    *,
    future_returned: bool,
    timeline_intersection_ms: float,
) -> str:
    if not future_returned:
        return "synchronous"
    if timeline_intersection_ms <= 0:
        return "not_overlapped"
    return "timeline_overlapped"


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
            raise InvalidOverlapMeasurement("timeline interval bounds are invalid")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def measurement_from_intervals(
    *,
    compute_interval: tuple[float, float],
    communication_intervals: Iterable[tuple[float, float]],
) -> OverlapMeasurement:
    return measurement_from_interval_sets(
        compute_intervals=(compute_interval,),
        communication_intervals=communication_intervals,
    )


def measurement_from_interval_sets(
    *,
    compute_intervals: Iterable[tuple[float, float]],
    communication_intervals: Iterable[tuple[float, float]],
) -> OverlapMeasurement:
    compute = merge_intervals(compute_intervals)
    communication = merge_intervals(communication_intervals)
    communication_ms = sum(end - start for start, end in communication)
    compute_ms = sum(end - start for start, end in compute)
    intersection_ms = sum(
        max(0.0, min(comm_end, compute_end) - max(comm_start, compute_start))
        for comm_start, comm_end in communication
        for compute_start, compute_end in compute
    )
    return OverlapMeasurement(
        communication_ms=communication_ms,
        compute_ms=compute_ms,
        overlapped_ms=communication_ms + compute_ms - intersection_ms,
        exposed_communication_ms=communication_ms - intersection_ms,
    )


def mean_measurement(measurements: Iterable[OverlapMeasurement]) -> OverlapMeasurement:
    values = tuple(measurements)
    if not values:
        return OverlapMeasurement(0.0, 0.0, 0.0, 0.0)
    divisor = len(values)
    return OverlapMeasurement(
        communication_ms=sum(value.communication_ms for value in values) / divisor,
        compute_ms=sum(value.compute_ms for value in values) / divisor,
        overlapped_ms=sum(value.overlapped_ms for value in values) / divisor,
        exposed_communication_ms=(
            sum(value.exposed_communication_ms for value in values) / divisor
        ),
    )


@dataclass(slots=True)
class _BucketEvents:
    start: Any
    launch_end: Any | None = None
    end: Any | None = None


@dataclass(slots=True)
class _StepEvents:
    origin: Any
    compute_end: Any | None
    buckets: list[_BucketEvents]
    future_returned: bool = False


class CudaOverlapRecorder:
    """Record DDP backward and completed bucket intervals without host waits."""

    def __init__(self, *, torch: Any, enabled: bool, asynchronous: bool = False) -> None:
        self._torch = torch
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.asynchronous = bool(asynchronous)
        self._active: _StepEvents | None = None
        self._completed: list[_StepEvents] = []
        self._lock = Lock()

    def begin_backward(self) -> None:
        if not self.enabled:
            return
        origin = self._event()
        origin.record()
        with self._lock:
            if self._active is not None:
                raise RuntimeError("overlap recorder already has an active backward")
            self._active = _StepEvents(origin=origin, compute_end=None, buckets=[])

    def end_backward(self) -> None:
        if not self.enabled:
            return
        compute_end = self._event()
        compute_end.record()
        with self._lock:
            if self._active is None:
                raise RuntimeError("overlap recorder has no active backward")
            self._active.compute_end = compute_end
            self._completed.append(self._active)
            self._active = None

    def wrap_hook(self, hook: Any) -> Any:
        if not self.enabled:
            return hook

        @wraps(hook)
        def measured_hook(state: Any, bucket: Any) -> Any:
            start = self._event()
            start.record()
            bucket_events = _BucketEvents(start=start)
            with self._lock:
                active = self._active
                if active is not None:
                    active.buckets.append(bucket_events)
            nvtx = getattr(self._torch.cuda, "nvtx", None)
            push = getattr(nvtx, "range_push", None)
            pop = getattr(nvtx, "range_pop", None)
            if callable(push):
                push("ccdl.bucket.communication.launch")
            try:
                future = hook(state, bucket)
            finally:
                if callable(pop):
                    pop()
            launch_end = self._event()
            launch_end.record()
            bucket_events.launch_end = launch_end
            then = getattr(future, "then", None)
            if callable(then):
                with self._lock:
                    if active is not None and self.asynchronous:
                        active.future_returned = True

                def record_completion(_future: Any) -> None:
                    end = self._event()
                    end.record()
                    bucket_events.end = end

                then(record_completion)
            return future

        return measured_hook

    def collect(self) -> tuple[OverlapMeasurement, str]:
        if not self.enabled:
            return OverlapMeasurement(0.0, 0.0, 0.0, 0.0), "not_measured"
        self._torch.cuda.synchronize()
        with self._lock:
            steps = tuple(self._completed)
            self._completed.clear()
        measurements = []
        future_returned = False
        for step in steps:
            if step.compute_end is None:
                raise RuntimeError("overlap recorder observed an incomplete backward")
            compute_end = float(step.origin.elapsed_time(step.compute_end))
            communication_intervals = []
            ordered_buckets = []
            for bucket in step.buckets:
                if bucket.end is None:
                    continue
                start = float(step.origin.elapsed_time(bucket.start))
                duration = float(bucket.start.elapsed_time(bucket.end))
                communication_intervals.append(
                    (max(0.0, start), max(0.0, start + duration))
                )
                if bucket.launch_end is not None:
                    launch_end = float(step.origin.elapsed_time(bucket.launch_end))
                    ordered_buckets.append((max(0.0, start), max(0.0, launch_end)))
            ordered_buckets.sort()
            compute_intervals = []
            compute_cursor = 0.0
            for bucket_start, launch_end in ordered_buckets:
                if bucket_start > compute_cursor:
                    compute_intervals.append((compute_cursor, bucket_start))
                compute_cursor = max(compute_cursor, launch_end)
            if compute_end > compute_cursor:
                compute_intervals.append((compute_cursor, compute_end))
            measurements.append(
                measurement_from_interval_sets(
                    compute_intervals=compute_intervals,
                    communication_intervals=communication_intervals,
                )
            )
            future_returned = future_returned or step.future_returned
        measurement = mean_measurement(measurements)
        return measurement, classify_overlap(
            future_returned=future_returned,
            timeline_intersection_ms=measurement.intersection_ms,
        )

    def _event(self) -> Any:
        return self._torch.cuda.Event(enable_timing=True)
