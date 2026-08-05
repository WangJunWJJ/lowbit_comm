"""Stable JSON metrics emitted by the end-to-end training example."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import fmean, median


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    measured_steps: int
    elapsed_seconds: float
    step_latencies_ms: tuple[float, ...]
    overlap_efficiency: float = 0.0
    communication_ms: float = 0.0
    compute_ms: float = 0.0
    overlapped_ms: float = 0.0
    exposed_communication_ms: float = 0.0
    overlap_classification: str = "not_measured"

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_latencies_ms", tuple(self.step_latencies_ms))
        if self.measured_steps < 1:
            raise ValueError("measured_steps must be >= 1")
        if self.elapsed_seconds <= 0 or not isfinite(self.elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite and > 0")
        if len(self.step_latencies_ms) != self.measured_steps:
            raise ValueError("step_latencies_ms must contain one value per measured step")
        if any(value < 0 or not isfinite(value) for value in self.step_latencies_ms):
            raise ValueError("step latencies must be finite and >= 0")
        if not 0.0 <= self.overlap_efficiency <= 1.0:
            raise ValueError("overlap_efficiency must be in [0, 1]")
        for name in (
            "communication_ms",
            "compute_ms",
            "overlapped_ms",
            "exposed_communication_ms",
        ):
            value = getattr(self, name)
            if value < 0 or not isfinite(value):
                raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    peak_allocated_bytes: int

    def __post_init__(self) -> None:
        if self.peak_allocated_bytes < 0:
            raise ValueError("peak_allocated_bytes must be >= 0")


@dataclass(frozen=True, slots=True)
class CorrectnessMetrics:
    rank_parameters_consistent: bool
    max_parameter_difference: float
    finite_loss: bool


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    requested_mode: str
    effective_strategy: str
    capability: str
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    mode: str
    world_size: int
    global_batch_size: int
    parameter_count: int
    workload: dict[str, object]
    timing: TimingMetrics
    memory: MemoryMetrics
    losses: tuple[float, ...]
    correctness: CorrectnessMetrics
    execution: ExecutionMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "workload", dict(self.workload))
        object.__setattr__(self, "losses", tuple(self.losses))
        if not self.workload:
            raise ValueError("workload must be non-empty")
        if not self.losses or any(not isfinite(loss) for loss in self.losses):
            raise ValueError("losses must be finite and non-empty")

    def to_dict(self) -> dict[str, object]:
        latencies = self.timing.step_latencies_ms
        timing = {
            "measured_steps": self.timing.measured_steps,
            "elapsed_seconds": self.timing.elapsed_seconds,
            "throughput_samples_per_second": (
                self.timing.measured_steps
                * self.global_batch_size
                / self.timing.elapsed_seconds
            ),
            "mean_step_latency_ms": fmean(latencies),
            "median_step_latency_ms": median(latencies),
            "p50_step_latency_ms": _percentile(latencies, 0.50),
            "p95_step_latency_ms": _percentile(latencies, 0.95),
            "min_step_latency_ms": min(latencies),
            "max_step_latency_ms": max(latencies),
            "overlap_efficiency": self.timing.overlap_efficiency,
            "communication_ms": self.timing.communication_ms,
            "compute_ms": self.timing.compute_ms,
            "overlapped_ms": self.timing.overlapped_ms,
            "exposed_communication_ms": self.timing.exposed_communication_ms,
            "overlap_classification": self.timing.overlap_classification,
        }
        return {
            "schema_version": 2,
            "mode": self.mode,
            "workload": dict(self.workload),
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "parameter_count": self.parameter_count,
            "timing": timing,
            "memory": asdict(self.memory),
            "loss": {
                "initial": self.losses[0],
                "final": self.losses[-1],
                "delta": self.losses[-1] - self.losses[0],
                "samples": list(self.losses),
            },
            "correctness": asdict(self.correctness),
            "execution": asdict(self.execution),
        }


def _percentile(values: tuple[float, ...], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
