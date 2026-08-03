"""Compiled dynamic-shape compressed all-gather executors."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from threading import RLock
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError
from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.executor import ObjectIdentity
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor
from ccdl_comm.quantization.sizing import estimate_quantized_size
from ccdl_comm.work import CollectiveWork, bind_execution_work

from .metadata_packet import (
    METADATA_PACKET_NUMEL,
    METADATA_PACKET_PROTOCOL_VERSION,
    decode_metadata_packets,
    write_metadata_packet,
)


DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION = METADATA_PACKET_PROTOCOL_VERSION
TENSOR_METADATA_AUTO_ENABLED = True
_METADATA_PROTOCOLS = frozenset({"object_v1", "tensor_v1", "auto"})


@dataclass(frozen=True, slots=True)
class DynamicGatherMetadata:
    """Versioned rank-local metadata transmitted before the fixed payload."""

    protocol_version: int
    shape: tuple[int, ...]
    dtype: str
    payload_numel: int

    def to_wire(self) -> dict[str, object]:
        """Return a pickle-safe protocol object for ``all_gather_object``."""

        return {
            "protocol_version": self.protocol_version,
            "shape": self.shape,
            "dtype": self.dtype,
            "payload_numel": self.payload_numel,
        }


@dataclass(frozen=True, slots=True)
class DynamicGatherCacheKey:
    """Inputs whose changes require a different dynamic gather executor."""

    shape_class: tuple[int, ...]
    dtype: str
    world_size: int
    config: CompressionConfig
    metadata_protocol: str
    process_group: ObjectIdentity
    distributed: ObjectIdentity
    extension_status: ObjectIdentity
    quantize: ObjectIdentity
    dequantize: ObjectIdentity


class DynamicGatherExecutorCache:
    """Bounded LRU cache keyed by an explicit cross-rank shape class."""

    def __init__(self, *, max_entries: int = 32) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[
            DynamicGatherCacheKey, CudaDynamicGatherExecutor
        ] = OrderedDict()
        self._lock = RLock()

    def get_or_create(
        self,
        key: DynamicGatherCacheKey,
        factory: Callable[[], "CudaDynamicGatherExecutor"],
    ) -> "CudaDynamicGatherExecutor":
        """Return a cached executor or create it once under the cache lock."""

        if not isinstance(key, DynamicGatherCacheKey):
            raise TypeError("key must be a DynamicGatherCacheKey")
        if not callable(factory):
            raise TypeError("factory must be callable")
        with self._lock:
            executor = self._entries.get(key)
            if executor is not None:
                self._entries.move_to_end(key)
                return executor
            executor = factory()
            if not isinstance(executor, CudaDynamicGatherExecutor):
                raise TypeError("factory must return a CudaDynamicGatherExecutor")
            self._entries[key] = executor
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return executor

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class CudaDynamicGatherExecutor:
    """Gather dynamic tensors within a compile-time cross-rank capacity."""

    def __init__(
        self,
        *,
        shape_class: tuple[int, ...],
        config: CompressionConfig,
        dtype: str,
        dist: Any,
        torch: Any,
        world_size: int,
        group: Any | None,
        extension_status: Any | None,
        quantize: Callable[..., Any],
        dequantize: Callable[..., Any],
        metadata_protocol_requested: str,
        metadata_protocol: str,
        metadata_protocol_fallback_reason: str | None,
    ) -> None:
        self._shape_class = _validate_shape_class(shape_class)
        self._config = config
        self._dtype = _normalize_dtype(dtype)
        self._dist = dist
        self._torch = torch
        self._world_size = world_size
        self._group = group
        self._extension_status = extension_status
        self._quantize = quantize
        self._dequantize = dequantize
        self._metadata_protocol_requested = metadata_protocol_requested
        self._metadata_protocol = metadata_protocol
        self._metadata_protocol_fallback_reason = metadata_protocol_fallback_reason
        self._metadata_workspaces: dict[str, tuple[object, object, object]] = {}
        capacity_numel = _numel(self._shape_class)
        estimate = estimate_quantized_size(
            capacity_numel,
            dtype=self._dtype,
            config=config,
        )
        self._capacity_payload_numel = estimate.quantized_bytes
        self.execution_info = ExecutionInfo(
            requested_strategy="compressed_dynamic_all_gather",
            executed_strategy="compressed_dynamic_all_gather",
            backend="cuda",
            fallback_used=False,
            fallback_reason=None,
            stage_names=(),
            original_bytes=estimate.original_bytes,
            compressed_bytes=estimate.quantized_bytes,
            compression_ratio=estimate.compression_ratio or 1.0,
            workspace_cache_hit=False,
            async_capable=False,
            fast_path=f"cuda_compiled_dynamic_all_gather_{metadata_protocol}",
            details={
                "shape_class": self._shape_class,
                "dtype": self._dtype,
                "world_size": world_size,
                "metadata_protocol_version": DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION,
                "metadata_protocol_requested": metadata_protocol_requested,
                "metadata_protocol_executed": metadata_protocol,
                "metadata_protocol_fallback_reason": (
                    metadata_protocol_fallback_reason
                ),
            },
        )
        self.execution_counters = ExecutionCounters()

    @property
    def shape_class(self) -> tuple[int, ...]:
        return self._shape_class

    @property
    def config(self) -> CompressionConfig:
        return self._config

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def metadata_protocol_version(self) -> int:
        return DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION

    @property
    def metadata_protocol(self) -> str:
        """Return the metadata transport selected at compile time."""

        return self._metadata_protocol

    def run(self, tensor: object) -> CollectiveWork[list[object]]:
        """Gather one dynamic tensor without recomputing its capacity class."""

        self.execution_counters._record_run()
        try:
            shape = tuple(getattr(tensor, "shape", ()))
            _validate_runtime_shape(shape, self._shape_class)
            active_dtype = _normalize_dtype(str(getattr(tensor, "dtype", "")))
            if active_dtype != self._dtype:
                raise TypeError(
                    f"compiled dynamic gather dtype is {self._dtype!r}; "
                    f"received {active_dtype!r}"
                )
            payload = (
                tensor.new_empty((0,), dtype=self._torch.uint8)
                if _numel(shape) == 0
                else self._quantize(
                    tensor,
                    self._config,
                    extension_status=self._extension_status,
                )
            )
            payload_numel = int(payload.numel())
            expected_payload_numel = estimate_quantized_size(
                _numel(shape),
                dtype=self._dtype,
                config=self._config,
            ).quantized_bytes
            if payload_numel != expected_payload_numel:
                raise ValueError(
                    "quantized payload size does not match runtime tensor shape "
                    f"({payload_numel} != {expected_payload_numel})"
                )
            local_metadata = DynamicGatherMetadata(
                protocol_version=DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION,
                shape=shape,
                dtype=self._dtype,
                payload_numel=payload_numel,
            )
            wire_metadata = self._exchange_metadata(local_metadata, tensor)
            metadata = tuple(
                _parse_metadata(
                    item,
                    shape_class=self._shape_class,
                    dtype=self._dtype,
                    config=self._config,
                    capacity_payload_numel=self._capacity_payload_numel,
                )
                for item in wire_metadata
            )
            padded_payload = _pad_payload(
                payload,
                self._capacity_payload_numel,
                self._torch,
            )
            gathered = [
                padded_payload.new_empty((self._capacity_payload_numel,))
                for _ in range(self._world_size)
            ]
            self._dist.all_gather(
                gathered,
                padded_payload,
                group=self._group,
            )
            result = [
                self._restore(buffer, item, tensor)
                for buffer, item in zip(gathered, metadata, strict=True)
            ]
            return bind_execution_work(
                result,
                self.execution_info,
                self.execution_counters,
            )
        except BaseException:
            self.execution_counters._record_failed()
            raise

    def _exchange_metadata(
        self,
        local_metadata: DynamicGatherMetadata,
        tensor: object,
    ) -> tuple[object, ...]:
        if self._metadata_protocol == "object_v1":
            wire_metadata: list[object | None] = [None] * self._world_size
            self._dist.all_gather_object(
                wire_metadata,
                local_metadata.to_wire(),
                group=self._group,
            )
            return tuple(wire_metadata)
        host_packet, local_packet, gathered_packet = self._metadata_workspace(
            tensor
        )
        write_metadata_packet(
            host_packet,
            shape=local_metadata.shape,
            dtype=local_metadata.dtype,
            payload_numel=local_metadata.payload_numel,
        )
        local_packet.copy_(host_packet, non_blocking=True)
        self._dist.all_gather_into_tensor(
            gathered_packet,
            local_packet,
            group=self._group,
        )
        return decode_metadata_packets(
            gathered_packet,
            world_size=self._world_size,
        )

    def _metadata_workspace(
        self,
        tensor: object,
    ) -> tuple[object, object, object]:
        device = getattr(tensor, "device", None)
        device_key = str(device)
        workspace = self._metadata_workspaces.get(device_key)
        if workspace is not None:
            return workspace
        host_packet = self._torch.empty(
            METADATA_PACKET_NUMEL,
            dtype=self._torch.int64,
            device="cpu",
            pin_memory=True,
        )
        local_packet = tensor.new_empty(
            (METADATA_PACKET_NUMEL,),
            dtype=self._torch.int64,
        )
        gathered_packet = tensor.new_empty(
            (self._world_size * METADATA_PACKET_NUMEL,),
            dtype=self._torch.int64,
        )
        workspace = (host_packet, local_packet, gathered_packet)
        self._metadata_workspaces[device_key] = workspace
        return workspace

    def _restore(
        self,
        buffer: object,
        metadata: DynamicGatherMetadata,
        template: object,
    ) -> object:
        if metadata.payload_numel == 0:
            return template.new_empty(metadata.shape)
        trimmed = buffer[: metadata.payload_numel]
        return self._dequantize(
            trimmed,
            metadata.shape,
            self._config,
            dtype=metadata.dtype,
            extension_status=self._extension_status,
            reduce_op="none",
        )


def compile_dynamic_all_gather(
    *,
    shape_class: tuple[int, ...],
    config: CompressionConfig,
    dtype: str,
    group: Any | None = None,
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize: Callable[..., Any] = dequantize_tensor,
    cache: DynamicGatherExecutorCache | None = None,
    metadata_protocol: str = "object_v1",
) -> CudaDynamicGatherExecutor:
    """Compile or retrieve one bounded dynamic all-gather executor."""

    if not isinstance(config, CompressionConfig):
        raise TypeError("config must be a CompressionConfig")
    active_shape_class = _validate_shape_class(shape_class)
    active_dtype = _normalize_dtype(dtype)
    dist = _distributed(import_module_fn)
    torch = import_module_fn("torch")
    world_size = int(dist.get_world_size(group=group))
    (
        active_metadata_protocol,
        metadata_protocol_fallback_reason,
    ) = _resolve_metadata_protocol(metadata_protocol, dist)
    key = DynamicGatherCacheKey(
        shape_class=active_shape_class,
        dtype=active_dtype,
        world_size=world_size,
        config=config,
        metadata_protocol=metadata_protocol,
        process_group=ObjectIdentity(group),
        distributed=ObjectIdentity(dist),
        extension_status=ObjectIdentity(extension_status),
        quantize=ObjectIdentity(quantize),
        dequantize=ObjectIdentity(dequantize),
    )

    def factory() -> CudaDynamicGatherExecutor:
        return CudaDynamicGatherExecutor(
            shape_class=active_shape_class,
            config=config,
            dtype=active_dtype,
            dist=dist,
            torch=torch,
            world_size=world_size,
            group=group,
            extension_status=extension_status,
            quantize=quantize,
            dequantize=dequantize,
            metadata_protocol_requested=metadata_protocol,
            metadata_protocol=active_metadata_protocol,
            metadata_protocol_fallback_reason=metadata_protocol_fallback_reason,
        )

    return factory() if cache is None else cache.get_or_create(key, factory)


def _resolve_metadata_protocol(
    requested: str,
    dist: Any,
) -> tuple[str, str | None]:
    if requested not in _METADATA_PROTOCOLS:
        raise ValueError(
            "metadata_protocol must be one of "
            f"{sorted(_METADATA_PROTOCOLS)!r}; received {requested!r}"
        )
    tensor_collective_available = callable(
        getattr(dist, "all_gather_into_tensor", None)
    )
    if requested == "tensor_v1":
        if not tensor_collective_available:
            raise RuntimeError(
                "tensor_v1 metadata requires torch.distributed."
                "all_gather_into_tensor"
            )
        return "tensor_v1", None
    if requested == "object_v1":
        return "object_v1", None
    if TENSOR_METADATA_AUTO_ENABLED and tensor_collective_available:
        return "tensor_v1", None
    reason = (
        "tensor_v1 performance gate is not approved"
        if tensor_collective_available
        else "all_gather_into_tensor is unavailable"
    )
    return "object_v1", reason


def _parse_metadata(
    value: object,
    *,
    shape_class: tuple[int, ...],
    dtype: str,
    config: CompressionConfig,
    capacity_payload_numel: int,
) -> DynamicGatherMetadata:
    if not isinstance(value, Mapping):
        raise RuntimeError("dynamic all-gather metadata must be a mapping")
    version = value.get("protocol_version")
    if version != DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION:
        raise RuntimeError(
            "dynamic all-gather metadata protocol version mismatch: "
            f"expected {DYNAMIC_GATHER_METADATA_PROTOCOL_VERSION}, received {version!r}"
        )
    shape_value = value.get("shape")
    if not isinstance(shape_value, (tuple, list)):
        raise RuntimeError("dynamic all-gather metadata shape must be a sequence")
    shape = tuple(shape_value)
    _validate_runtime_shape(shape, shape_class)
    metadata_dtype = value.get("dtype")
    if metadata_dtype != dtype:
        raise RuntimeError(
            f"dynamic all-gather metadata dtype must be {dtype!r}; "
            f"received {metadata_dtype!r}"
        )
    payload_numel = value.get("payload_numel")
    if (
        isinstance(payload_numel, bool)
        or not isinstance(payload_numel, int)
        or payload_numel < 0
        or payload_numel > capacity_payload_numel
    ):
        raise RuntimeError(
            "dynamic all-gather metadata payload_numel is outside the "
            "compiled shape class capacity"
        )
    expected_payload_numel = estimate_quantized_size(
        _numel(shape),
        dtype=dtype,
        config=config,
    ).quantized_bytes
    if payload_numel != expected_payload_numel:
        raise RuntimeError(
            "dynamic all-gather metadata payload size does not match shape: "
            f"{payload_numel} != {expected_payload_numel}"
        )
    return DynamicGatherMetadata(
        protocol_version=version,
        shape=shape,
        dtype=dtype,
        payload_numel=payload_numel,
    )


def _pad_payload(payload: object, target_numel: int, torch: Any) -> object:
    payload_numel = int(payload.numel())
    if payload_numel == target_numel:
        return payload
    zeros = payload.new_zeros((target_numel - payload_numel,))
    return torch.cat((payload, zeros), dim=0)


def _distributed(import_module_fn: Callable[[str], Any]) -> Any:
    try:
        dist = import_module_fn("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError(
            "torch.distributed is not available"
        ) from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError(
            "torch.distributed is not initialized"
        )
    return dist


def _validate_shape_class(shape_class: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(shape_class)
    if not shape:
        raise ValueError("shape_class must contain at least one dimension")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        raise ValueError("shape_class dimensions must be non-negative integers")
    return shape


def _validate_runtime_shape(
    shape: tuple[int, ...],
    shape_class: tuple[int, ...],
) -> None:
    valid = len(shape) == len(shape_class) and all(
        isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and 0 <= dimension <= bound
        for dimension, bound in zip(shape, shape_class, strict=True)
    )
    if not valid:
        raise ValueError(
            f"tensor shape {shape} exceeds compiled shape class capacity "
            f"{shape_class}"
        )


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower().removeprefix("torch.")
    resolved = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }.get(normalized, normalized)
    if resolved not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unsupported dynamic all-gather dtype: {dtype!r}")
    return resolved


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)
