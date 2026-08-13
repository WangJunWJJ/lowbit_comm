"""Backend-independent value, output, wire, and algorithm types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class ErrorFeedbackDomain(Enum):
    NONE = "none"
    GRADIENT = "gradient"
    PARAMETER_DELTA = "parameter_delta"


@dataclass(frozen=True, slots=True)
class FullTensor:
    dtype: DataType

    def __post_init__(self) -> None:
        _require_dtype(self.dtype)


@dataclass(frozen=True, slots=True)
class ReducedShard:
    dtype: DataType
    layout_version: int

    def __post_init__(self) -> None:
        _require_dtype(self.dtype)
        if isinstance(self.layout_version, bool) or not isinstance(self.layout_version, int):
            raise TypeError("layout_version must be an integer")
        if self.layout_version < 0:
            raise ValueError("layout_version must be >= 0")


@dataclass(frozen=True, slots=True)
class QuantizedWire:
    bit: int
    group_size: int
    quant_type: str = "linear"
    compact: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.bit, bool) or not isinstance(self.bit, int):
            raise TypeError("bit must be an integer")
        if self.bit not in {4, 8}:
            raise ValueError("bit must be 4 or 8")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int):
            raise TypeError("group_size must be an integer")
        if self.group_size not in {16, 32, 64}:
            raise ValueError("group_size must be 16, 32, or 64")
        if not isinstance(self.quant_type, str) or not self.quant_type.strip():
            raise ValueError("quant_type must be a non-empty string")
        if not isinstance(self.compact, bool):
            raise TypeError("compact must be a bool")


@dataclass(frozen=True, slots=True)
class FullPrecisionWire:
    dtype: DataType

    def __post_init__(self) -> None:
        _require_dtype(self.dtype)


@dataclass(frozen=True, slots=True)
class AutoAlgorithm:
    """Request compile-time evidence-gated algorithm selection."""


@dataclass(frozen=True, slots=True)
class NativeAllReduce:
    """Use the backend-native all-reduce implementation."""


@dataclass(frozen=True, slots=True)
class CompressedAllGather:
    """Gather compressed full contributions and reduce locally."""


@dataclass(frozen=True, slots=True)
class CompressedReduceScatter:
    """Return a globally reduced rank-local shard."""


@dataclass(frozen=True, slots=True)
class CompressedReduceScatterAllGather:
    """Keep reduce-scatter and final full-tensor gather quantized."""


@dataclass(frozen=True, slots=True)
class HierarchicalCompressed:
    """Reduce and distribute quantized values through bounded rank groups."""

    max_fan_in: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.max_fan_in, bool) or not isinstance(self.max_fan_in, int):
            raise TypeError("max_fan_in must be an integer")
        if self.max_fan_in <= 1:
            raise ValueError("max_fan_in must be greater than one")


def _require_dtype(value: object) -> None:
    if not isinstance(value, DataType):
        raise TypeError("dtype must be a DataType")
