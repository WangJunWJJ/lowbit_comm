"""Deterministic CUDA payload and workspace layouts."""

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


@dataclass(frozen=True, slots=True)
class ReducedShardLayout:
    """Exact ownership and byte layout for one ReducedShard exchange."""

    global_numel: int
    logical_shard_length: int
    transport_shard_length: int
    offset: int
    valid_length: int
    group_size: int | None
    groups_per_shard: int
    payload_bytes_per_destination: int
    send_payload_bytes: int
    receive_payload_bytes: int
    output_numel: int
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


def build_reduced_shard_layout(
    *,
    numel: int,
    dtype: str,
    world_size: int,
    compression: CompressionKind,
    group_size: int | None,
    rank: int,
) -> ReducedShardLayout:
    """Build one rank's validated ReducedShard ownership and byte layout."""
    if type(numel) is not int or numel < 0:
        raise CompileError(
            "CUDA layout numel must be a non-negative integer."
        )
    if type(dtype) is not str or dtype not in _SUPPORTED_DTYPES:
        raise CompileError("CUDA ReducedShard dtype is unsupported.")
    if (
        type(world_size) is not int
        or world_size <= 0
        or world_size > _MAX_LAYOUT_VALUE
    ):
        raise CompileError("CUDA ReducedShard world size is invalid.")
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise CompileError("CUDA ReducedShard rank is outside world size.")
    if type(compression) is not CompressionKind:
        raise CompileError("CUDA ReducedShard compression is invalid.")

    shard_numerator = _checked_add(
        numel,
        world_size - 1,
        "logical shard",
    )
    logical_shard_length = shard_numerator // world_size
    padded_input_numel = _checked_mul(
        logical_shard_length,
        world_size,
        "padded input",
    )
    offset = min(
        _checked_mul(rank, logical_shard_length, "shard offset"),
        numel,
    )
    valid_length = min(logical_shard_length, numel - offset)
    output_bytes = _checked_mul(
        logical_shard_length,
        _OUTPUT_BYTES_PER_ELEMENT,
        "output",
    )

    if compression is CompressionKind.NONE:
        if group_size is not None:
            raise CompileError(
                "CUDA native layout cannot set a group size."
            )
        workspace_bytes = 0
        if padded_input_numel != numel:
            workspace_bytes = _checked_mul(
                padded_input_numel,
                _OUTPUT_BYTES_PER_ELEMENT,
                "padded input",
            )
        return ReducedShardLayout(
            global_numel=numel,
            logical_shard_length=logical_shard_length,
            transport_shard_length=logical_shard_length,
            offset=offset,
            valid_length=valid_length,
            group_size=None,
            groups_per_shard=0,
            payload_bytes_per_destination=0,
            send_payload_bytes=0,
            receive_payload_bytes=0,
            output_numel=logical_shard_length,
            output_bytes=output_bytes,
            workspace_bytes=workspace_bytes,
        )
    if compression is not CompressionKind.INT8:
        raise CompileError("CUDA ReducedShard compression is unsupported.")
    if (
        type(group_size) is not int
        or group_size not in _SUPPORTED_GROUP_SIZES
    ):
        raise CompileError("CUDA INT8 group size is unsupported.")

    groups_numerator = _checked_add(
        logical_shard_length,
        group_size - 1,
        "transport shard",
    )
    groups_per_shard = groups_numerator // group_size
    transport_shard_length = _checked_mul(
        groups_per_shard,
        group_size,
        "transport shard",
    )
    bytes_per_group = _checked_add(group_size, 2, "payload")
    payload_bytes_per_destination = _checked_mul(
        groups_per_shard,
        bytes_per_group,
        "payload",
    )
    send_payload_bytes = _checked_mul(
        payload_bytes_per_destination,
        world_size,
        "send payload",
    )
    receive_payload_bytes = _checked_mul(
        payload_bytes_per_destination,
        world_size,
        "receive payload",
    )
    workspace_bytes = _checked_add(
        send_payload_bytes,
        receive_payload_bytes,
        "workspace",
    )
    return ReducedShardLayout(
        global_numel=numel,
        logical_shard_length=logical_shard_length,
        transport_shard_length=transport_shard_length,
        offset=offset,
        valid_length=valid_length,
        group_size=group_size,
        groups_per_shard=groups_per_shard,
        payload_bytes_per_destination=payload_bytes_per_destination,
        send_payload_bytes=send_payload_bytes,
        receive_payload_bytes=receive_payload_bytes,
        output_numel=logical_shard_length,
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


__all__ = [
    "FullTensorLayout",
    "ReducedShardLayout",
    "build_fulltensor_layout",
    "build_reduced_shard_layout",
]
