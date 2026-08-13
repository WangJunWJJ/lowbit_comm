"""Typed CUDA point-to-point transport primitives."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import Condition
from typing import Any, Callable

from lowbit_comm.core import DataType, MetadataPacket, QuantizedWire
from lowbit_comm.core.metadata import METADATA_PACKET_WORDS
from lowbit_comm.runtime import CompletionManager, CompletionPipeline, CompletionWork

from .codec import dequantize_into, payload_nbytes, quantize_into
from .loader import CudaExtensionStatus


@dataclass(frozen=True, slots=True)
class P2PTags:
    """Physical tags reserved by one logical quantized P2P operation."""

    metadata: int
    payload: int

    @classmethod
    def from_logical(cls, tag: int) -> "P2PTags":
        if isinstance(tag, bool) or not isinstance(tag, int):
            raise TypeError("tag must be an integer")
        if tag < 0:
            raise ValueError("tag must be non-negative")
        return cls(metadata=2 * tag, payload=2 * tag + 1)


class _HandlesEvent:
    def __init__(self, handles: tuple[object, ...]) -> None:
        self._handles = handles

    def query(self) -> bool:
        return all(_handle_done(handle) for handle in self._handles)

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        for handle in self._handles:
            handle.wait()
        return True


class CudaQuantizedSender:
    """Reusable typed sender with dynamic-shape device metadata."""

    def __init__(
        self,
        *,
        peer: int,
        tag: int,
        dtype: DataType,
        wire: QuantizedWire,
        extension_status: CudaExtensionStatus,
        process_group: object | None = None,
        torch: Any | None = None,
        dist: Any | None = None,
        quantize: Callable[..., object] = quantize_into,
    ) -> None:
        if isinstance(peer, bool) or not isinstance(peer, int) or peer < 0:
            raise ValueError("peer must be a non-negative integer")
        if not isinstance(dtype, DataType):
            raise TypeError("dtype must be a DataType")
        if not isinstance(wire, QuantizedWire):
            raise TypeError("wire must be a QuantizedWire")
        self._peer = peer
        self._tags = P2PTags.from_logical(tag)
        self._dtype = dtype
        self._wire = wire
        self._status = extension_status
        self._group = process_group
        self._torch = torch or import_module("torch")
        self._dist = dist or import_module("torch.distributed")
        self._quantize = quantize

    def isend(
        self,
        tensor: object,
        *,
        layout_generation: int = 0,
        flags: int = 0,
    ) -> CompletionWork[None]:
        shape = tuple(int(dimension) for dimension in tensor.shape)  # type: ignore[attr-defined]
        original_numel = int(tensor.numel())  # type: ignore[attr-defined]
        prepared = _pad_source(tensor, original_numel, self._wire.group_size)
        payload_size = payload_nbytes(
            original_numel,
            dtype=self._dtype,
            wire=self._wire,
        )
        payload = self._torch.empty(
            payload_size,
            dtype=self._torch.uint8,
            device=tensor.device,  # type: ignore[attr-defined]
        )
        self._quantize(
            prepared,
            payload,
            self._wire,
            extension_status=self._status,
        )
        packet = MetadataPacket(
            shape=shape,
            dtype=self._dtype,
            wire=self._wire,
            payload_numel=payload_size,
            layout_generation=layout_generation,
            flags=flags,
        )
        metadata = self._torch.tensor(
            packet.to_values(),
            dtype=self._torch.int64,
            device=tensor.device,  # type: ignore[attr-defined]
        )
        metadata_handle = self._dist.isend(
            metadata,
            self._peer,
            group=self._group,
            tag=self._tags.metadata,
        )
        payload_handle = self._dist.isend(
            payload,
            self._peer,
            group=self._group,
            tag=self._tags.payload,
        )
        resources = (tensor, prepared, metadata, payload)
        return CompletionWork(
            None,
            event=_HandlesEvent((metadata_handle, payload_handle)),
            resources=resources,
        )

    def send(
        self,
        tensor: object,
        *,
        layout_generation: int = 0,
        flags: int = 0,
    ) -> None:
        self.isend(
            tensor,
            layout_generation=layout_generation,
            flags=flags,
        ).wait()


_P2P_COMPLETION_MANAGER = CompletionManager()


class CudaQuantizedReceiver:
    """Typed dynamic receiver whose host decode runs off the caller thread."""

    def __init__(
        self,
        *,
        peer: int,
        tag: int,
        dtype: DataType,
        wire: QuantizedWire,
        extension_status: CudaExtensionStatus,
        device: object,
        process_group: object | None = None,
        torch: Any | None = None,
        dist: Any | None = None,
        dequantize: Callable[..., object] = dequantize_into,
    ) -> None:
        if isinstance(peer, bool) or not isinstance(peer, int) or peer < 0:
            raise ValueError("peer must be a non-negative integer")
        if not isinstance(dtype, DataType):
            raise TypeError("dtype must be a DataType")
        if not isinstance(wire, QuantizedWire):
            raise TypeError("wire must be a QuantizedWire")
        self._peer = peer
        self._tags = P2PTags.from_logical(tag)
        self._dtype = dtype
        self._wire = wire
        self._status = extension_status
        self._device = device
        self._group = process_group
        self._torch = torch or import_module("torch")
        self._dist = dist or import_module("torch.distributed")
        self._dequantize = dequantize

    def irecv(self) -> CompletionPipeline[object | None]:
        metadata = self._torch.empty(
            METADATA_PACKET_WORDS,
            dtype=self._torch.int64,
            device=self._device,
        )
        metadata_handle = self._dist.irecv(
            metadata,
            self._peer,
            group=self._group,
            tag=self._tags.metadata,
        )
        payload_ready = _DeferredEvent()
        output_ready = _DeferredEvent()
        pipeline: CompletionPipeline[object | None] = CompletionPipeline(None)
        pipeline.add_stage(
            "metadata",
            _HandlesEvent((metadata_handle,)),
            action=lambda _result: self._submit_payload(
                metadata,
                payload_ready,
                output_ready,
            ),
        )
        pipeline.add_stage(
            "payload",
            payload_ready,
            action=self._dequantize_pending,
        )
        pipeline.add_stage("output", output_ready)
        _P2P_COMPLETION_MANAGER.submit(pipeline)
        return pipeline

    def recv(self) -> object:
        return self.irecv().wait()

    def _submit_payload(
        self,
        metadata: object,
        payload_ready: "_DeferredEvent",
        output_ready: "_DeferredEvent",
    ) -> object:
        torch = self._torch
        with torch.cuda.device(self._device):
            packet = MetadataPacket.from_values(metadata.tolist())  # type: ignore[attr-defined]
            if packet.dtype is not self._dtype:
                raise TypeError(
                    f"received dtype {packet.dtype.value}; expected {self._dtype.value}"
                )
            if packet.wire != self._wire:
                raise ValueError(
                    f"received quantized wire {packet.wire}; expected {self._wire}"
                )
            expected_payload = payload_nbytes(
                packet.logical_numel,
                dtype=packet.dtype,
                wire=packet.wire,
            )
            if packet.payload_numel != expected_payload:
                raise ValueError(
                    "metadata payload length does not match tensor and quant schema"
                )
            payload = torch.empty(
                packet.payload_numel,
                dtype=torch.uint8,
                device=self._device,
            )
            payload_handle = self._dist.irecv(
                payload,
                self._peer,
                group=self._group,
                tag=self._tags.payload,
            )
            payload_ready.bind(_HandlesEvent((payload_handle,)))
            return _PendingReceive(packet, payload, output_ready)

    def _dequantize_pending(
        self,
        pending: object | None,
    ) -> object:
        if not isinstance(pending, _PendingReceive):
            raise TypeError("payload stage requires pending receive state")
        packet = pending.packet
        payload = pending.payload
        torch = self._torch
        with torch.cuda.device(self._device):
            padded_numel = (
                (packet.logical_numel + packet.wire.group_size - 1)
                // packet.wire.group_size
                * packet.wire.group_size
            )
            output = torch.empty(
                padded_numel,
                dtype=_torch_dtype(torch, packet.dtype),
                device=self._device,
            )
            self._dequantize(
                payload,
                output,
                packet.wire,
                dtype=packet.dtype,
                extension_status=self._status,
            )
            result = output[: packet.logical_numel].reshape(packet.shape)
            event = torch.cuda.Event(enable_timing=False)
            event.record(torch.cuda.current_stream(self._device))
            pending.output_ready.bind(_CudaEvent(event))
            return result


@dataclass(frozen=True, slots=True)
class _PendingReceive:
    packet: MetadataPacket
    payload: object
    output_ready: "_DeferredEvent"


class _DeferredEvent:
    def __init__(self) -> None:
        self._event: object | None = None
        self._condition = Condition()

    def bind(self, event: object) -> None:
        with self._condition:
            if self._event is not None:
                raise RuntimeError("deferred event is already bound")
            self._event = event
            self._condition.notify_all()

    def query(self) -> bool:
        with self._condition:
            event = self._event
        return bool(event.query()) if event is not None else False  # type: ignore[attr-defined]

    def wait(self, timeout: float | None = None) -> bool:
        with self._condition:
            if self._event is None and not self._condition.wait_for(
                lambda: self._event is not None,
                timeout,
            ):
                return False
            event = self._event
        return bool(event.wait(timeout))  # type: ignore[attr-defined]


class _CudaEvent:
    def __init__(self, event: object) -> None:
        self._event = event

    def query(self) -> bool:
        return bool(self._event.query())  # type: ignore[attr-defined]

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            raise NotImplementedError("CUDA event does not support timeout")
        self._event.synchronize()  # type: ignore[attr-defined]
        return True


def _pad_source(tensor: object, numel: int, group_size: int) -> object:
    if numel % group_size == 0:
        reshape = getattr(tensor, "reshape", None)
        return reshape(-1) if callable(reshape) else tensor
    flat = tensor.reshape(-1)  # type: ignore[attr-defined]
    padded_numel = ((numel + group_size - 1) // group_size) * group_size
    padded = flat.new_zeros((padded_numel,))
    padded[:numel].copy_(flat)
    return padded


def _handle_done(handle: object) -> bool:
    query = getattr(handle, "is_completed", None)
    if callable(query):
        return bool(query())
    query = getattr(handle, "query", None)
    return bool(query()) if callable(query) else False


def _torch_dtype(torch: Any, dtype: DataType) -> object:
    return {
        DataType.FP16: torch.float16,
        DataType.BF16: torch.bfloat16,
        DataType.FP32: torch.float32,
    }[dtype]
