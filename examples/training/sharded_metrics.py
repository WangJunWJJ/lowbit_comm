"""Stable phase metrics for the sharded SGD training example."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean
from typing import Mapping


PHASE_NAMES = (
    "backward_and_flatten",
    "compressed_reduce_scatter",
    "local_shard_update",
    "parameter_all_gather",
    "parameter_writeback",
)


@dataclass(frozen=True, slots=True)
class ShardedPhaseMetrics:
    measured_steps: int
    samples_ms: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        if self.measured_steps < 1:
            raise ValueError("measured_steps must be >= 1")
        samples = {name: tuple(values) for name, values in self.samples_ms.items()}
        if set(samples) != set(PHASE_NAMES):
            raise ValueError("samples_ms must contain exactly the supported phase names")
        for name, values in samples.items():
            if len(values) != self.measured_steps:
                raise ValueError(f"{name} must contain one sample per measured step")
            if any(not isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} samples must be finite and >= 0")
        object.__setattr__(self, "samples_ms", samples)

    def to_dict(self) -> dict[str, float]:
        return {name: fmean(self.samples_ms[name]) for name in PHASE_NAMES}


def augment_training_payload(
    payload: Mapping[str, object],
    *,
    mode: str,
    phases: ShardedPhaseMetrics,
    phases_measured: bool,
    initial_pointers: Mapping[str, int],
    final_pointers: Mapping[str, int],
) -> dict[str, object]:
    """Add sharded phase and buffer evidence without mutating base metrics."""

    initial = dict(initial_pointers)
    final = dict(final_pointers)
    if initial != final:
        raise ValueError("buffer pointers changed during measured training")
    if not isinstance(phases_measured, bool):
        raise TypeError("phases_measured must be a boolean")
    result = dict(payload)
    execution = dict(result.get("execution", {}))
    execution["requested_mode"] = mode
    result.update(
        schema_version=3,
        mode=mode,
        execution=execution,
        phase_timing_ms=phases.to_dict(),
        phase_timing_measured=phases_measured,
        buffer_reuse={
            "stable": True,
            "initial_pointers": initial,
            "final_pointers": final,
        },
    )
    return result
