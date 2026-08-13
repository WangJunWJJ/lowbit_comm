"""Allocation-free facade over the reviewed native CUDA codec ABI."""

from __future__ import annotations

from math import ceil
from typing import Any

from lowbit_comm.core import DataType, QuantizedWire
from lowbit_comm.core.errors import LowBitCommError

from .loader import CudaExtensionStatus, load_cuda_extension


class ExtensionUnavailable(LowBitCommError):
    """The optional native codec cannot satisfy the requested operation."""


_DTYPE_BYTES = {
    DataType.FP16: 2,
    DataType.BF16: 2,
    DataType.FP32: 4,
}

_QUANT_TYPE_NAMES = {
    "linear": "Linear",
    "normal": "Normal",
    "uniform": "Uniform",
    "e3m0": "E3M0",
    "e2m1": "E2M1",
}


def payload_nbytes(numel: int, *, dtype: DataType, wire: QuantizedWire) -> int:
    """Return exact bytes for the native compact grouped payload."""

    if isinstance(numel, bool) or not isinstance(numel, int) or numel < 0:
        raise ValueError("numel must be a non-negative integer")
    if not isinstance(dtype, DataType):
        raise TypeError("dtype must be a DataType")
    if not isinstance(wire, QuantizedWire):
        raise TypeError("wire must be a QuantizedWire")
    groups = ceil(numel / wire.group_size) if numel else 0
    value_bytes = wire.group_size * wire.bit // 8
    scale_bytes = _DTYPE_BYTES[dtype]
    return groups * (value_bytes + scale_bytes)


def quantize_into(
    source: object,
    output: object,
    wire: QuantizedWire,
    *,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Quantize into a caller-owned payload buffer."""

    module = _require_module(extension_status)
    native = _require_symbol(module, "inplace_quantize")
    native(
        source,
        output,
        wire.group_size,
        0,
        False,
        wire.bit,
        _quant_type(module, wire),
        wire.compact,
    )
    return output


def quantize_chunks_into(
    source: object,
    output: object,
    wire: QuantizedWire,
    *,
    chunk_numel: int,
    chunks: int,
    payload_stride: int,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Quantize contiguous destination chunks with one native dispatch."""

    for name, value in (
        ("chunk_numel", chunk_numel),
        ("chunks", chunks),
        ("payload_stride", payload_stride),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    module = _require_module(extension_status)
    native = _require_symbol(module, "inplace_quantize_chunks")
    used = native(
        source,
        output,
        chunk_numel,
        chunks,
        payload_stride,
        wire.group_size,
        0,
        False,
        wire.bit,
        _quant_type(module, wire),
        wire.compact,
    )
    if not used:
        raise ExtensionUnavailable("native chunk quantization declined the wire schema")
    return output


def dequantize_into(
    payload: object,
    output: object,
    wire: QuantizedWire,
    *,
    dtype: DataType,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Dequantize one payload into a caller-owned floating-point buffer."""

    if not isinstance(dtype, DataType):
        raise TypeError("dtype must be a DataType")
    module = _require_module(extension_status)
    native = _require_symbol(module, "inplace_dequantize")
    reduce_op = _require_symbol(_require_symbol(module, "ReduceOP"), "NONE")
    native(
        payload,
        output,
        wire.group_size,
        0,
        wire.bit,
        reduce_op,
        _quant_type(module, wire),
        wire.compact,
    )
    return output


def decode_dynamic_metadata_into(
    metadata: object,
    descriptors: object,
    *,
    world_size: int,
    dtype: DataType,
    wire: QuantizedWire,
    layout_generation: int,
    max_numel: int,
    payload_stride: int,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Validate fixed metadata packets and emit compact device descriptors."""

    module = _require_module(extension_status)
    native = _require_symbol(module, "inplace_decode_dynamic_metadata")
    native(
        metadata,
        descriptors,
        world_size,
        _dtype_code(dtype),
        wire.bit,
        wire.group_size,
        _quant_type_code(wire.quant_type),
        wire.compact,
        layout_generation,
        max_numel,
        payload_stride,
    )
    return descriptors


def _require_module(status: CudaExtensionStatus | None) -> object:
    active = status or load_cuda_extension()
    if not active.available or active.module is None:
        raise ExtensionUnavailable(active.reason or "CUDA extension is unavailable")
    return active.module


def _require_symbol(owner: object, name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError as error:
        raise ExtensionUnavailable(f"CUDA extension missing required symbol: {name}") from error


def _quant_type(module: object, wire: QuantizedWire) -> object:
    try:
        name = _QUANT_TYPE_NAMES[wire.quant_type]
    except KeyError as error:
        raise ValueError(f"unsupported quant_type: {wire.quant_type!r}") from error
    return _require_symbol(_require_symbol(module, "QuantType"), name)


def _dtype_code(dtype: DataType) -> int:
    return {DataType.FP16: 1, DataType.BF16: 2, DataType.FP32: 3}[dtype]


def _quant_type_code(quant_type: str) -> int:
    return {"linear": 1, "normal": 2, "uniform": 3, "e3m0": 4, "e2m1": 5}[
        quant_type
    ]
