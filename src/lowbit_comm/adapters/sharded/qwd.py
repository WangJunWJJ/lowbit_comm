"""Quantized weight-difference preparation and conservative policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ParameterCommunicationDecision:
    mode: Literal["fp_refresh", "qwd"]
    bit: Literal[8]
    reason: str


class SafeInt8QWDPolicy:
    def __init__(
        self,
        *,
        warmup_steps: int = 100,
        refresh_interval: int = 512,
        relative_error_threshold: float = 1.0e-2,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if not isfinite(relative_error_threshold) or relative_error_threshold <= 0:
            raise ValueError("relative_error_threshold must be finite and positive")
        self.warmup_steps = warmup_steps
        self.refresh_interval = refresh_interval
        self.relative_error_threshold = float(relative_error_threshold)

    def decide(
        self,
        *,
        step: int,
        relative_error: float | None,
        capability: bool,
        sensitive: bool = False,
    ) -> ParameterCommunicationDecision:
        if step <= 0:
            raise ValueError("step must be positive")
        if step <= self.warmup_steps:
            return _decision("fp_refresh", "warmup")
        if sensitive:
            return _decision("fp_refresh", "sensitive_tensor")
        if not capability:
            return _decision("fp_refresh", "capability")
        if relative_error is not None and relative_error > self.relative_error_threshold:
            return _decision("fp_refresh", "error_threshold")
        if step % self.refresh_interval == 0:
            return _decision("fp_refresh", "periodic_refresh")
        return _decision("qwd", "int8_qwd")


def prepare_parameter_delta(master: Any, model_shard: Any, *, valid_numel: int) -> Any:
    """Return FP32 ``master - model`` and zero its padded suffix."""

    if int(master.numel()) != int(model_shard.numel()):
        raise ValueError("master and model shard sizes must match")
    if valid_numel < 0 or valid_numel > int(master.numel()):
        raise ValueError("valid_numel must be within the shard")
    if "float32" not in str(master.dtype):
        raise TypeError("master shard must use float32")
    delta = master - model_shard.float()
    if valid_numel < int(delta.numel()):
        delta[valid_numel:].zero_()
    return delta


def apply_quantized_weight_difference(
    model_shard: Any, restored_delta: Any, *, valid_numel: int
) -> Any:
    """Apply lossy qWD by addition so the next delta carries residual error."""

    _validate_writeback(model_shard, restored_delta, valid_numel)
    model_shard[:valid_numel].add_(restored_delta[:valid_numel])
    return model_shard


def full_precision_refresh(
    model_shard: Any, master_shard: Any, *, valid_numel: int
) -> Any:
    """Overwrite a model shard from the authoritative FP32 master shard."""

    _validate_writeback(model_shard, master_shard, valid_numel)
    model_shard[:valid_numel].copy_(master_shard[:valid_numel])
    return model_shard


def _validate_writeback(model: Any, source: Any, valid_numel: int) -> None:
    if int(model.numel()) != int(source.numel()):
        raise ValueError("writeback shard sizes must match")
    if valid_numel < 0 or valid_numel > int(model.numel()):
        raise ValueError("valid_numel must be within the shard")


def _decision(
    mode: Literal["fp_refresh", "qwd"], reason: str
) -> ParameterCommunicationDecision:
    return ParameterCommunicationDecision(mode, 8, reason)
