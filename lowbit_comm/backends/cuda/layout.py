"""Deterministic FullTensor INT8 payload and workspace layouts."""

from __future__ import annotations

from dataclasses import dataclass

from lowbit_comm.api.policy import CompressionKind
from lowbit_comm.core.errors import CompileError


_MAX_LAYOUT_VALUE = (1 << 63) - 1
_SUPPORTED_DTYPES = frozenset({"fp16", "bf16"})
_SUPPORTED_WORLD_SIZES = frozenset({2, 4})
_SUPPORTED_GROUP_SIZES = frozenset({16, 32, 64})
_SCALE_BYTES_PER_GROUP = 2
_OUTPUT_BYTES_PER_ELEMENT = 2


@dataclass(frozen=True, slots=True)
class FullTensorLayout:
    """Exact byte layout for one compact INT8 FullTensor exchange."""

    logical_numel: int
    padded_numel: int
    group_size: int | None
    group_count: int
    payload_bytes_per_rank: int
    gathered_payload_bytes: int
    output_bytes: int
    workspace_bytes: int


def build_fulltensor_layout(
    *,
    numel: int,
    dtype: str,
    world_size: int,
    compression: CompressionKind,
    group_size: int | None,
) -> FullTensorLayout:
    """Build a validated, immutable FullTensor execution layout.

    Compact INT8 contributes one byte per padded element and one scale stored
    in the source FP16/BF16 dtype per group. Native execution has no quantized
    payload or workspace. Sizes use the signed 64-bit descriptor boundary.
    """
    if type(numel) is not int or numel < 0:
        raise CompileError(
            "CUDA layout numel must be a non-negative integer."
        )
    if type(dtype) is not str or dtype not in _SUPPORTED_DTYPES:
        raise CompileError("CUDA FullTensor dtype is unsupported.")
    if (
        type(world_size) is not int
        or world_size not in _SUPPORTED_WORLD_SIZES
    ):
        raise CompileError("CUDA FullTensor world size is unsupported.")
    if type(compression) is not CompressionKind:
        raise CompileError("CUDA FullTensor compression is invalid.")
    output_bytes = _checked_mul(
        numel,
        _OUTPUT_BYTES_PER_ELEMENT,
        "output",
    )
    if compression is CompressionKind.NONE:
        if group_size is not None:
            raise CompileError("CUDA native layout cannot set a group size.")
        return FullTensorLayout(
            logical_numel=numel,
            padded_numel=numel,
            group_size=None,
            group_count=0,
            payload_bytes_per_rank=0,
            gathered_payload_bytes=0,
            output_bytes=output_bytes,
            workspace_bytes=0,
        )
    if compression is not CompressionKind.INT8:
        raise CompileError("CUDA FullTensor compression is unsupported.")
    if (
        type(group_size) is not int
        or group_size not in _SUPPORTED_GROUP_SIZES
    ):
        raise CompileError("CUDA INT8 group size is unsupported.")

    groups_numerator = _checked_add(
        numel,
        group_size - 1,
        "group count",
    )
    group_count = groups_numerator // group_size
    padded_numel = _checked_mul(group_count, group_size, "padded numel")
    scale_bytes = _checked_mul(
        group_count,
        _SCALE_BYTES_PER_GROUP,
        "scale metadata",
    )
    payload_bytes_per_rank = _checked_add(
        padded_numel,
        scale_bytes,
        "payload",
    )
    gathered_payload_bytes = _checked_mul(
        payload_bytes_per_rank,
        world_size,
        "gathered payload",
    )
    workspace_bytes = _checked_mul(
        payload_bytes_per_rank,
        world_size + 1,
        "workspace payload",
    )

    return FullTensorLayout(
        logical_numel=numel,
        padded_numel=padded_numel,
        group_size=group_size,
        group_count=group_count,
        payload_bytes_per_rank=payload_bytes_per_rank,
        gathered_payload_bytes=gathered_payload_bytes,
        output_bytes=output_bytes,
        workspace_bytes=workspace_bytes,
    )


def _checked_add(left: int, right: int, label: str) -> int:
    result = left + right
    if result > _MAX_LAYOUT_VALUE:
        raise CompileError(f"CUDA FullTensor {label} size overflow.")
    return result


def _checked_mul(left: int, right: int, label: str) -> int:
    result = left * right
    if result > _MAX_LAYOUT_VALUE:
        raise CompileError(f"CUDA FullTensor {label} size overflow.")
    return result


__all__ = ["FullTensorLayout", "build_fulltensor_layout"]
