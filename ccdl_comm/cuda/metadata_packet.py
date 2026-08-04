"""Fixed-layout tensor metadata packets for dynamic CUDA communication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


METADATA_PACKET_PROTOCOL_VERSION = 1
METADATA_PACKET_MAX_NDIM = 8
METADATA_PACKET_DIMS_OFFSET = 8
METADATA_PACKET_NUMEL = METADATA_PACKET_DIMS_OFFSET + METADATA_PACKET_MAX_NDIM
METADATA_PACKET_RESERVED_OFFSET = 5
_METADATA_PACKET_RESERVED_END = METADATA_PACKET_DIMS_OFFSET

_DTYPE_TO_CODE = {
    "fp16": 1,
    "bf16": 2,
    "fp32": 3,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def encode_metadata_packet(
    *,
    shape: Sequence[int],
    dtype: str,
    payload_numel: int,
    torch: Any,
    device: object,
    flags: int = 0,
    output: object | None = None,
) -> object:
    """Encode dynamic tensor metadata into one fixed-layout int64 tensor.

    Args:
        shape: Runtime tensor shape. Up to eight dimensions are supported.
        dtype: Canonical CCDL dtype name.
        payload_numel: Number of valid bytes in the quantized payload.
        torch: Torch-compatible module used to construct the tensor lazily.
        device: Device on which the packet must reside.
        flags: Non-negative protocol flags reserved for compatible extensions.
        output: Optional caller-owned fixed-size int64 tensor to overwrite.

    Returns:
        The caller-owned output when provided, otherwise a new packet tensor.

    Raises:
        TypeError: If metadata fields are not integer sequences.
        ValueError: If metadata exceeds the fixed protocol capacity.
    """

    packet = (
        torch.empty(METADATA_PACKET_NUMEL, dtype=torch.int64, device=device)
        if output is None
        else output
    )
    _validate_output(packet, torch)
    return write_metadata_packet(
        packet,
        shape=shape,
        dtype=dtype,
        payload_numel=payload_numel,
        flags=flags,
    )


def write_metadata_packet(
    output: object,
    *,
    shape: Sequence[int],
    dtype: str,
    payload_numel: int,
    flags: int = 0,
) -> object:
    """Write metadata into an existing fixed-layout packet without allocation."""

    active_shape = _validate_shape(shape)
    dtype_code = _DTYPE_TO_CODE.get(dtype)
    if dtype_code is None:
        raise ValueError(f"unsupported metadata dtype: {dtype!r}")
    _require_non_negative_integer(payload_numel, "payload_numel")
    _require_non_negative_integer(flags, "flags")
    values = [0] * METADATA_PACKET_NUMEL
    values[0] = METADATA_PACKET_PROTOCOL_VERSION
    values[1] = len(active_shape)
    values[2] = dtype_code
    values[3] = payload_numel
    values[4] = flags
    values[
        METADATA_PACKET_DIMS_OFFSET : METADATA_PACKET_DIMS_OFFSET
        + len(active_shape)
    ] = active_shape
    for index, value in enumerate(values):
        output[index] = value
    return output


def decode_metadata_packet(packet: object) -> Mapping[str, object]:
    """Validate and decode one metadata packet into canonical host values.

    Calling tolist() on a CUDA packet establishes the host visibility needed
    to construct variable-shape result tensors. Callers should do this only
    after the metadata collective has completed.

    Args:
        packet: Fixed-size int64 tensor-like packet.

    Returns:
        A mapping containing protocol version, shape, dtype, payload size and
        flags.

    Raises:
        RuntimeError: If the packet violates the versioned wire protocol.
    """

    try:
        values = packet.tolist()
    except AttributeError as exc:
        raise RuntimeError("metadata packet must expose tolist()") from exc
    return _decode_metadata_values(values)


def decode_metadata_packets(
    gathered_packet: object,
    *,
    world_size: int,
) -> tuple[Mapping[str, object], ...]:
    """Decode a contiguous rank-major packet tensor with one host read."""

    _require_non_negative_integer(world_size, "world_size")
    if world_size == 0:
        raise ValueError("world_size must be positive")
    try:
        values = gathered_packet.tolist()
    except AttributeError as exc:
        raise RuntimeError("gathered metadata packet must expose tolist()") from exc
    expected = world_size * METADATA_PACKET_NUMEL
    if not isinstance(values, list) or len(values) != expected:
        raise RuntimeError(
            f"gathered metadata packet must contain {expected} int64 values"
        )
    return tuple(
        _decode_metadata_values(
            values[
                rank * METADATA_PACKET_NUMEL : (rank + 1)
                * METADATA_PACKET_NUMEL
            ]
        )
        for rank in range(world_size)
    )


def _decode_metadata_values(values: object) -> Mapping[str, object]:
    if not isinstance(values, list) or len(values) != METADATA_PACKET_NUMEL:
        raise RuntimeError(
            f"metadata packet must contain {METADATA_PACKET_NUMEL} int64 values"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError("metadata packet values must be integers")
    version = values[0]
    if version != METADATA_PACKET_PROTOCOL_VERSION:
        raise RuntimeError(
            "metadata packet protocol version mismatch: "
            f"expected {METADATA_PACKET_PROTOCOL_VERSION}, received {version}"
        )
    ndim = values[1]
    if ndim < 0 or ndim > METADATA_PACKET_MAX_NDIM:
        raise RuntimeError("metadata packet rank exceeds the fixed maximum rank")
    dtype = _CODE_TO_DTYPE.get(values[2])
    if dtype is None:
        raise RuntimeError(f"metadata packet has unknown dtype code {values[2]}")
    payload_numel = values[3]
    flags = values[4]
    if payload_numel < 0:
        raise RuntimeError("metadata packet payload_numel must be non-negative")
    if flags < 0:
        raise RuntimeError("metadata packet flags must be non-negative")
    if any(values[METADATA_PACKET_RESERVED_OFFSET:_METADATA_PACKET_RESERVED_END]):
        raise RuntimeError("metadata packet reserved fields must be zero")
    shape_values = values[
        METADATA_PACKET_DIMS_OFFSET : METADATA_PACKET_DIMS_OFFSET
        + METADATA_PACKET_MAX_NDIM
    ]
    shape = tuple(shape_values[:ndim])
    if any(dimension < 0 for dimension in shape):
        raise RuntimeError("metadata packet shape dimensions must be non-negative")
    if any(shape_values[ndim:]):
        raise RuntimeError("metadata packet unused shape fields must be zero")
    return {
        "protocol_version": version,
        "shape": shape,
        "dtype": dtype,
        "payload_numel": payload_numel,
        "flags": flags,
    }


def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
    try:
        active_shape = tuple(shape)
    except TypeError as exc:
        raise TypeError("shape must be an integer sequence") from exc
    if len(active_shape) > METADATA_PACKET_MAX_NDIM:
        raise ValueError(
            "metadata shape exceeds maximum rank "
            f"{METADATA_PACKET_MAX_NDIM}: {len(active_shape)}"
        )
    for dimension in active_shape:
        _require_non_negative_integer(dimension, "shape dimension")
    return active_shape


def _require_non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_output(output: object, torch: Any) -> None:
    if getattr(output, "dtype", None) != torch.int64:
        raise ValueError("metadata packet output must use torch.int64")
    if int(output.numel()) != METADATA_PACKET_NUMEL:
        raise ValueError(
            f"metadata packet output must contain {METADATA_PACKET_NUMEL} values"
        )


__all__ = [
    "METADATA_PACKET_DIMS_OFFSET",
    "METADATA_PACKET_MAX_NDIM",
    "METADATA_PACKET_NUMEL",
    "METADATA_PACKET_PROTOCOL_VERSION",
    "METADATA_PACKET_RESERVED_OFFSET",
    "decode_metadata_packet",
    "decode_metadata_packets",
    "encode_metadata_packet",
    "write_metadata_packet",
]
