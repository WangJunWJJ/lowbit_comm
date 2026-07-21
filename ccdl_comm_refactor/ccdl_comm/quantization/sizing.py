from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ccdl_comm.config import CompressionConfig


_DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
}


@dataclass(frozen=True)
class QuantizationSizeEstimate:
    """Estimated storage footprint for one quantized tensor buffer."""

    numel: int
    padded_numel: int
    padding_numel: int
    num_groups: int
    original_bytes: int
    quantized_bytes: int
    compression_ratio: float


def _dtype_size(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype={dtype!r}; expected one of {sorted(_DTYPE_BYTES)}") from exc


def _metadata_bytes_per_group(dtype: str, topk: int) -> int:
    dtype_bytes = _dtype_size(dtype)
    metadata = dtype_bytes
    if dtype == "fp32":
        if topk == 1:
            metadata += 8
        elif topk == 2:
            metadata += 12
    else:
        if topk == 1:
            metadata += 4
        elif topk == 2:
            metadata += 6
    return metadata


def estimate_quantized_size(
    numel: int,
    *,
    dtype: str,
    config: CompressionConfig,
) -> QuantizationSizeEstimate:
    """Estimate quantized buffer size for ParaScale planning.

    Args:
        numel: Number of logical tensor elements before padding.
        dtype: Logical source dtype: `fp16`, `bf16`, or `fp32`.
        config: Compression policy.

    Returns:
        A size estimate including group padding and per-group metadata.

    Raises:
        ValueError: If `numel` is negative or `dtype` is unsupported.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")

    dtype_bytes = _dtype_size(dtype)
    num_groups = ceil(numel / config.group_size) if numel else 0
    padded_numel = num_groups * config.group_size
    quantized_value_bytes = config.group_size * config.bit // 8
    bytes_per_group = quantized_value_bytes + _metadata_bytes_per_group(dtype, config.topk)
    original_bytes = numel * dtype_bytes
    quantized_bytes = num_groups * bytes_per_group
    compression_ratio = original_bytes / quantized_bytes if quantized_bytes else 0.0
    return QuantizationSizeEstimate(
        numel=numel,
        padded_numel=padded_numel,
        padding_numel=padded_numel - numel,
        num_groups=num_groups,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        compression_ratio=compression_ratio,
    )
