"""Weight-difference metadata, preparation, and safe communication policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ParameterDeltaShard:
    """A rank-local FP32 master-minus-model parameter difference."""

    shard: Any
    shard_index: int
    shard_numel: int
    valid_numel: int
    original_numel: int
    padded_numel: int
    world_size: int
    layout_version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.shard_index, "shard_index")
        _require_nonnegative_integer(self.shard_numel, "shard_numel")
        _require_nonnegative_integer(self.valid_numel, "valid_numel")
        _require_nonnegative_integer(self.original_numel, "original_numel")
        _require_nonnegative_integer(self.padded_numel, "padded_numel")
        _require_positive_integer(self.world_size, "world_size")
        _require_nonnegative_integer(self.layout_version, "layout_version")
        if self.shard_index >= self.world_size:
            raise ValueError("shard_index must be smaller than world_size")
        if self.valid_numel > self.shard_numel:
            raise ValueError("valid_numel must be <= shard_numel")
        if self.padded_numel != self.shard_numel * self.world_size:
            raise ValueError("padded_numel must equal shard_numel * world_size")
        if self.original_numel > self.padded_numel:
            raise ValueError("original_numel must be <= padded_numel")
        expected_valid = max(
            0,
            min(
                self.shard_numel,
                self.original_numel - self.shard_index * self.shard_numel,
            ),
        )
        if self.valid_numel != expected_valid:
            raise ValueError("valid_numel does not match the rank-local logical range")
        if _tensor_numel(self.shard, "delta shard") != self.shard_numel:
            raise ValueError("delta shard numel must equal shard_numel")
        _require_contiguous(self.shard, "delta shard")
        if _canonical_dtype(self.shard) != "fp32":
            raise ValueError("delta shard must use fp32")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ParameterCommunicationDecision:
    """One deterministic qWD or full-precision refresh decision."""

    mode: Literal["fp_refresh", "qwd"]
    bit: Literal[8]
    reason: str

    def __post_init__(self) -> None:
        if self.mode not in {"fp_refresh", "qwd"}:
            raise ValueError("mode must be fp_refresh or qwd")
        if isinstance(self.bit, bool) or self.bit != 8:
            raise ValueError("bit must be 8")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TypeError("reason must be a non-empty string")


@runtime_checkable
class ParameterDeltaProvider(Protocol):
    """Prepare a caller-owned FP32 parameter-difference workspace."""

    def prepare_delta(
        self,
        master_shard: Any,
        model_shard: Any,
        *,
        out: Any,
        valid_numel: int,
    ) -> Any: ...


@runtime_checkable
class ParameterCommunicationPolicy(Protocol):
    """Select qWD or refresh without executing tensor operations."""

    def decide(
        self,
        *,
        step: int,
        tensor_role: str,
        numel: int,
        relative_error: float | None,
        capability: bool,
    ) -> ParameterCommunicationDecision: ...


class TorchParameterDeltaProvider:
    """Compute an FP32 master-minus-model delta without importing torch."""

    def prepare_delta(
        self,
        master_shard: Any,
        model_shard: Any,
        *,
        out: Any,
        valid_numel: int,
    ) -> Any:
        """Write the valid FP32 difference and zero any padded suffix."""

        _require_nonnegative_integer(valid_numel, "valid_numel")
        master_numel = _tensor_numel(master_shard, "master shard")
        if valid_numel > master_numel:
            raise ValueError("valid_numel must be <= shard numel")
        for name, tensor in (("model shard", model_shard), ("output", out)):
            if _tensor_numel(tensor, name) != master_numel:
                raise ValueError(f"{name} numel must match master shard")
        for name, tensor in (
            ("master shard", master_shard),
            ("model shard", model_shard),
            ("output", out),
        ):
            _require_contiguous(tensor, name)
        if _canonical_dtype(master_shard) != "fp32":
            raise ValueError("master shard must use fp32")
        if _canonical_dtype(out) != "fp32":
            raise ValueError("output must use fp32")
        if _canonical_dtype(model_shard) not in {"fp16", "bf16", "fp32"}:
            raise ValueError("model shard must use fp16, bf16, or fp32")
        master_device = getattr(master_shard, "device", None)
        if getattr(model_shard, "device", None) != master_device:
            raise ValueError("model shard device must match master shard")
        if getattr(out, "device", None) != master_device:
            raise ValueError("output device must match master shard")
        copy = getattr(out, "copy_", None)
        subtract = getattr(out, "sub_", None)
        narrow = getattr(out, "narrow", None)
        if not callable(copy) or not callable(subtract) or not callable(narrow):
            raise TypeError("output must expose copy_(), sub_(), and narrow()")
        copy(master_shard)
        subtract(model_shard)
        padding_numel = master_numel - valid_numel
        if padding_numel:
            padding = narrow(0, valid_numel, padding_numel)
            zero = getattr(padding, "zero_", None)
            if not callable(zero):
                raise TypeError("output slice must expose zero_()")
            zero()
        return out


class SafeInt8QWDPolicy:
    """Conservative deterministic policy for INT8 qWD parameter traffic."""

    def __init__(
        self,
        *,
        warmup_steps: int = 100,
        refresh_interval: int = 512,
        relative_error_threshold: float = 1.0e-2,
        error_check_interval: int = 128,
    ) -> None:
        _require_nonnegative_integer(warmup_steps, "warmup_steps")
        _require_positive_integer(refresh_interval, "refresh_interval")
        _require_positive_integer(error_check_interval, "error_check_interval")
        self._warmup_steps = warmup_steps
        self._refresh_interval = refresh_interval
        self._relative_error_threshold = _require_positive_finite(
            relative_error_threshold,
            "relative_error_threshold",
        )
        self._error_check_interval = error_check_interval

    @property
    def warmup_steps(self) -> int:
        return self._warmup_steps

    @property
    def refresh_interval(self) -> int:
        return self._refresh_interval

    @property
    def relative_error_threshold(self) -> float:
        return self._relative_error_threshold

    @property
    def error_check_interval(self) -> int:
        return self._error_check_interval

    def configuration_packet(self) -> tuple[int, int, int, int]:
        """Return a stable integer representation for cross-rank validation."""

        return (
            self._warmup_steps,
            self._refresh_interval,
            int(round(self._relative_error_threshold * 1_000_000_000)),
            self._error_check_interval,
        )

    def decide(
        self,
        *,
        step: int,
        tensor_role: str,
        numel: int,
        relative_error: float | None,
        capability: bool,
    ) -> ParameterCommunicationDecision:
        """Select a rank-deterministic action from validated scalar inputs."""

        _require_positive_integer(step, "step")
        _require_positive_integer(numel, "numel")
        if not isinstance(tensor_role, str) or not tensor_role.strip():
            raise TypeError("tensor_role must be a non-empty string")
        if not isinstance(capability, bool):
            raise TypeError("capability must be a boolean")
        if relative_error is not None:
            relative_error = _require_nonnegative_finite(
                relative_error,
                "relative_error",
            )
        if step <= self._warmup_steps:
            return _decision("fp_refresh", "warmup")
        if tensor_role == "sensitive":
            return _decision("fp_refresh", "sensitive_tensor")
        if not capability:
            return _decision("fp_refresh", "capability")
        if (
            relative_error is not None
            and relative_error > self._relative_error_threshold
        ):
            return _decision("fp_refresh", "error_threshold")
        if step % self._refresh_interval == 0:
            return _decision("fp_refresh", "periodic_refresh")
        return _decision("qwd", "int8_qwd")


def _decision(
    mode: Literal["fp_refresh", "qwd"],
    reason: str,
) -> ParameterCommunicationDecision:
    return ParameterCommunicationDecision(mode=mode, bit=8, reason=reason)


def _canonical_dtype(tensor: Any) -> str:
    value = str(getattr(tensor, "dtype", "")).lower()
    return {
        "float16": "fp16",
        "torch.float16": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "bf16": "bf16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "fp32": "fp32",
    }.get(value, value)


def _tensor_numel(tensor: Any, name: str) -> int:
    numel = getattr(tensor, "numel", None)
    if not callable(numel):
        raise TypeError(f"{name} must expose numel()")
    return int(numel())


def _require_contiguous(tensor: Any, name: str) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(is_contiguous) or not bool(is_contiguous()):
        raise ValueError(f"{name} must be contiguous")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_integer(value: object, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _require_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


__all__ = [
    "ParameterCommunicationDecision",
    "ParameterCommunicationPolicy",
    "ParameterDeltaProvider",
    "ParameterDeltaShard",
    "SafeInt8QWDPolicy",
    "TorchParameterDeltaProvider",
]
