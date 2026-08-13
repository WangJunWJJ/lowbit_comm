"""Backend-independent fixed-layout metadata for dynamic communication."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Sequence

from .types import DataType, QuantizedWire


METADATA_PROTOCOL_VERSION = 1
METADATA_PACKET_MAX_NDIM = 8
METADATA_PACKET_WORDS = 24
_DIMS_OFFSET = 16
_RESERVED = slice(11, _DIMS_OFFSET)

_DTYPE_TO_CODE = {DataType.FP16: 1, DataType.BF16: 2, DataType.FP32: 3}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_QUANT_TO_CODE = {"linear": 1, "normal": 2, "uniform": 3, "e3m0": 4, "e2m1": 5}
_CODE_TO_QUANT = {code: name for name, code in _QUANT_TO_CODE.items()}


@dataclass(frozen=True, slots=True)
class MetadataPacket:
    """Versioned metadata whose encoded length never depends on tensor shape."""

    shape: tuple[int, ...]
    dtype: DataType
    wire: QuantizedWire
    payload_numel: int
    layout_generation: int
    flags: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape(self.shape))
        if not isinstance(self.dtype, DataType):
            raise TypeError("dtype must be a DataType")
        if not isinstance(self.wire, QuantizedWire):
            raise TypeError("wire must be a QuantizedWire")
        if self.wire.quant_type not in _QUANT_TO_CODE:
            raise ValueError(f"unsupported quant_type: {self.wire.quant_type!r}")
        _non_negative(self.payload_numel, "payload_numel")
        _non_negative(self.layout_generation, "layout_generation")
        _non_negative(self.flags, "flags")

    @property
    def logical_numel(self) -> int:
        return reduce(mul, self.shape, 1)

    def to_values(self) -> tuple[int, ...]:
        values = [0] * METADATA_PACKET_WORDS
        values[0] = METADATA_PROTOCOL_VERSION
        values[1] = len(self.shape)
        values[2] = _DTYPE_TO_CODE[self.dtype]
        values[3] = self.payload_numel
        values[4] = self.wire.bit
        values[5] = self.wire.group_size
        values[6] = _QUANT_TO_CODE[self.wire.quant_type]
        values[7] = int(self.wire.compact)
        values[8] = self.layout_generation
        values[9] = self.flags
        values[10] = self.logical_numel
        values[_DIMS_OFFSET : _DIMS_OFFSET + len(self.shape)] = self.shape
        return tuple(values)

    @classmethod
    def from_values(cls, packet: Sequence[int]) -> "MetadataPacket":
        values = tuple(packet)
        if len(values) != METADATA_PACKET_WORDS:
            raise ValueError(
                f"metadata packet must contain {METADATA_PACKET_WORDS} integers"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("metadata packet values must be integers")
        if values[0] != METADATA_PROTOCOL_VERSION:
            raise ValueError(
                "metadata packet protocol version mismatch: "
                f"expected {METADATA_PROTOCOL_VERSION}, received {values[0]}"
            )
        ndim = values[1]
        if ndim < 0 or ndim > METADATA_PACKET_MAX_NDIM:
            raise ValueError("metadata packet rank exceeds the fixed maximum rank")
        if any(values[_RESERVED]):
            raise ValueError("metadata packet reserved words must be zero")
        shape_words = values[_DIMS_OFFSET:]
        if any(shape_words[ndim:]):
            raise ValueError("metadata packet unused shape words must be zero")
        try:
            dtype = _CODE_TO_DTYPE[values[2]]
        except KeyError as error:
            raise ValueError(f"unknown dtype code: {values[2]}") from error
        try:
            quant_type = _CODE_TO_QUANT[values[6]]
        except KeyError as error:
            raise ValueError(f"unknown quant_type code: {values[6]}") from error
        if values[7] not in (0, 1):
            raise ValueError("metadata packet compact field must be zero or one")
        result = cls(
            shape=tuple(shape_words[:ndim]),
            dtype=dtype,
            wire=QuantizedWire(
                bit=values[4],
                group_size=values[5],
                quant_type=quant_type,
                compact=bool(values[7]),
            ),
            payload_numel=values[3],
            layout_generation=values[8],
            flags=values[9],
        )
        if result.logical_numel != values[10]:
            raise ValueError("metadata packet logical_numel does not match shape")
        return result


def _shape(value: object) -> tuple[int, ...]:
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("shape must be an integer sequence") from error
    if len(result) > METADATA_PACKET_MAX_NDIM:
        raise ValueError(
            f"metadata shape exceeds maximum rank {METADATA_PACKET_MAX_NDIM}"
        )
    for dimension in result:
        _non_negative(dimension, "shape dimension")
    return result


def _non_negative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
