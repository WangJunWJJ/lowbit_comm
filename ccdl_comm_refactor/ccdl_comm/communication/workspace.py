from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import ceil
from operator import mul
from typing import Any, Callable

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.codec import allocate_dequantized_buffer


@dataclass(frozen=True)
class _WorkspaceMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str
    padded_numel: int


@dataclass
class _WorkspaceRecord:
    metadata: _WorkspaceMetadata
    tensor: Any


class DequantizedWorkspaceCache:
    """Per-hook cache for restored dequantization workspace tensors."""

    def __init__(
        self,
        *,
        allocator: Callable[[Any, tuple[int, ...], CompressionConfig], Any] = allocate_dequantized_buffer,
    ) -> None:
        self._allocator = allocator
        self._records: dict[Any, _WorkspaceRecord] = {}

    def get(self, key: Any, tensor: Any, shape: tuple[int, ...], config: CompressionConfig) -> Any:
        metadata = _metadata_for(tensor, shape, config)
        record = self._records.get(key)
        if record is not None and record.metadata == metadata:
            return record.tensor
        workspace = self._allocator(tensor, shape, config)
        self._records[key] = _WorkspaceRecord(metadata=metadata, tensor=workspace)
        return workspace

    def clear(self) -> None:
        self._records.clear()


def _metadata_for(tensor: Any, shape: tuple[int, ...], config: CompressionConfig) -> _WorkspaceMetadata:
    return _WorkspaceMetadata(
        shape=tuple(shape),
        dtype=str(getattr(tensor, "dtype", "")),
        device=str(getattr(tensor, "device", "")),
        padded_numel=_padded_numel(shape, config.group_size),
    )


def _padded_numel(shape: tuple[int, ...], group_size: int) -> int:
    numel = reduce(mul, shape, 1)
    if numel == 0:
        return 0
    return ceil(numel / group_size) * group_size
