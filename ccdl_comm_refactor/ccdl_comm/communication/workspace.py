from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import reduce
from math import ceil
from operator import mul
from typing import Any, Callable

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.codec import allocate_dequantized_buffer, allocate_quantized_buffer


@dataclass(frozen=True)
class _WorkspaceMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str
    padded_numel: int
    estimated_bytes: int


@dataclass
class _WorkspaceRecord:
    metadata: _WorkspaceMetadata
    tensor: Any


@dataclass
class _ShardWorkspaceRecord:
    tensor: Any
    estimated_bytes: int


class DequantizedWorkspaceCache:
    """Per-hook cache for restored dequantization workspace tensors."""

    def __init__(
        self,
        *,
        allocator: Callable[[Any, tuple[int, ...], CompressionConfig], Any] = allocate_dequantized_buffer,
        max_entries: int | None = None,
        max_cached_bytes: int | None = None,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None")
        if max_cached_bytes is not None and max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be >= 0 or None")
        self._allocator = allocator
        self._max_entries = max_entries
        self._max_cached_bytes = max_cached_bytes
        self._records: OrderedDict[Any, _WorkspaceRecord] = OrderedDict()
        self._cached_bytes = 0

    def get(self, key: Any, tensor: Any, shape: tuple[int, ...], config: CompressionConfig) -> Any:
        metadata = _metadata_for(tensor, shape, config)
        record = self._records.get(key)
        if record is not None and record.metadata == metadata:
            self._records.move_to_end(key)
            return record.tensor
        if record is not None:
            self._cached_bytes -= record.metadata.estimated_bytes
        workspace = self._allocator(tensor, shape, config)
        self._records[key] = _WorkspaceRecord(metadata=metadata, tensor=workspace)
        self._records.move_to_end(key)
        self._cached_bytes += metadata.estimated_bytes
        self._evict_over_budget()
        return workspace

    def clear(self) -> None:
        self._records.clear()
        self._cached_bytes = 0

    def _evict_over_budget(self) -> None:
        while self._max_entries is not None and len(self._records) > self._max_entries:
            _key, record = self._records.popitem(last=False)
            self._cached_bytes -= record.metadata.estimated_bytes
        while self._max_cached_bytes is not None and self._cached_bytes > self._max_cached_bytes and self._records:
            _key, record = self._records.popitem(last=False)
            self._cached_bytes -= record.metadata.estimated_bytes


class ShardCommunicationWorkspaceCache:
    """Cache send, receive, and reduced-shard workspaces for shard collectives."""

    def __init__(
        self,
        *,
        quantized_allocator: Callable[[Any, CompressionConfig, str], Any] | None = None,
        reduced_allocator: Callable[[Any, tuple[int, ...], CompressionConfig], Any] = allocate_dequantized_buffer,
        max_entries: int | None = None,
        max_cached_bytes: int | None = None,
    ) -> None:
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be >= 1 or None")
        if max_cached_bytes is not None and max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be >= 0 or None")
        self._quantized_allocator = quantized_allocator or _allocate_quantized_workspace
        self._reduced_allocator = reduced_allocator
        self._max_entries = max_entries
        self._max_cached_bytes = max_cached_bytes
        self._records: OrderedDict[Any, _ShardWorkspaceRecord] = OrderedDict()
        self._cached_bytes = 0

    def get_quantized_chunk(
        self,
        bucket_key: Any,
        chunk_index: int,
        tensor: Any,
        config: CompressionConfig,
        *,
        dtype: str,
        world_size: int,
    ) -> Any:
        key = (
            "send",
            bucket_key,
            chunk_index,
            world_size,
            _tensor_signature(tensor),
            _compression_signature(config),
            dtype,
        )
        return self._get_or_allocate(key, lambda: self._quantized_allocator(tensor, config, dtype))

    def get_received_payload(
        self,
        bucket_key: Any,
        payload_template: Any,
        index: int,
        *,
        world_size: int,
        config: CompressionConfig,
    ) -> Any:
        key = (
            "recv",
            bucket_key,
            index,
            world_size,
            _tensor_signature(payload_template),
            _compression_signature(config),
        )
        return self._get_or_allocate(
            key,
            lambda: payload_template.new_empty(tuple(payload_template.shape)),
        )

    def get_reduced_shard(
        self,
        bucket_key: Any,
        tensor: Any,
        shape: tuple[int, ...],
        config: CompressionConfig,
        *,
        dtype: str,
        world_size: int,
        rank: int,
    ) -> Any:
        key = (
            "reduced",
            bucket_key,
            rank,
            world_size,
            tuple(shape),
            _tensor_signature(tensor),
            _compression_signature(config),
            dtype,
        )
        return self._get_or_allocate(key, lambda: self._reduced_allocator(tensor, shape, config))

    def clear(self) -> None:
        self._records.clear()
        self._cached_bytes = 0

    def _get_or_allocate(self, key: Any, allocate: Callable[[], Any]) -> Any:
        if key in self._records:
            self._records.move_to_end(key)
            return self._records[key].tensor
        workspace = allocate()
        record = _ShardWorkspaceRecord(
            tensor=workspace,
            estimated_bytes=_tensor_estimated_bytes(workspace),
        )
        self._records[key] = record
        self._records.move_to_end(key)
        self._cached_bytes += record.estimated_bytes
        self._evict_over_budget()
        return workspace

    def _evict_over_budget(self) -> None:
        while self._max_entries is not None and len(self._records) > self._max_entries:
            _key, record = self._records.popitem(last=False)
            self._cached_bytes -= record.estimated_bytes
        while self._max_cached_bytes is not None and self._cached_bytes > self._max_cached_bytes and self._records:
            _key, record = self._records.popitem(last=False)
            self._cached_bytes -= record.estimated_bytes


def _metadata_for(tensor: Any, shape: tuple[int, ...], config: CompressionConfig) -> _WorkspaceMetadata:
    padded_numel = _padded_numel(shape, config.group_size)
    return _WorkspaceMetadata(
        shape=tuple(shape),
        dtype=str(getattr(tensor, "dtype", "")),
        device=str(getattr(tensor, "device", "")),
        padded_numel=padded_numel,
        estimated_bytes=padded_numel * _dtype_size_bytes(getattr(tensor, "dtype", "")),
    )


def _padded_numel(shape: tuple[int, ...], group_size: int) -> int:
    numel = reduce(mul, shape, 1)
    if numel == 0:
        return 0
    return ceil(numel / group_size) * group_size


def _dtype_size_bytes(dtype: Any) -> int:
    dtype_name = str(dtype).lower()
    if "float64" in dtype_name or "double" in dtype_name:
        return 8
    if "float32" in dtype_name or dtype_name.endswith("float"):
        return 4
    if "float16" in dtype_name or "bfloat16" in dtype_name or dtype_name.endswith("half"):
        return 2
    if "int8" in dtype_name or "uint8" in dtype_name or "bool" in dtype_name:
        return 1
    return 4


def _allocate_quantized_workspace(tensor: Any, config: CompressionConfig, dtype: str) -> Any:
    return allocate_quantized_buffer(tensor, config, dtype=dtype)


def _tensor_signature(tensor: Any) -> tuple[Any, ...]:
    return (
        tuple(getattr(tensor, "shape", ())),
        str(getattr(tensor, "dtype", "")),
        str(getattr(tensor, "device", "")),
    )


def _tensor_estimated_bytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(tensor, "numel", None)
    count = int(numel()) if callable(numel) else reduce(mul, getattr(tensor, "shape", ()), 1)
    element_size = getattr(tensor, "element_size", None)
    if callable(element_size):
        return count * int(element_size())
    return count * _dtype_size_bytes(getattr(tensor, "dtype", ""))


def _compression_signature(config: CompressionConfig) -> tuple[Any, ...]:
    return (
        config.bit,
        config.group_size,
        config.topk,
        config.quant_type,
        config.compact,
    )
