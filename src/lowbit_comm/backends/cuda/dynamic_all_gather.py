"""Dynamic-shape quantized all-gather using fixed device metadata packets."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from lowbit_comm.core import DataType, MetadataPacket, QuantizedWire
from lowbit_comm.core.metadata import METADATA_PACKET_WORDS
from lowbit_comm.runtime import CompletionWork, ImmediateCompletionEvent

from .codec import dequantize_into, payload_nbytes, quantize_into
from .loader import CudaExtensionStatus


class CudaDynamicAllGather:
    """Gather differently shaped rank-local tensors over one quantized wire."""

    def __init__(
        self,
        *,
        dtype: DataType,
        wire: QuantizedWire,
        layout_generation: int,
        max_numel: int,
        extension_status: CudaExtensionStatus | object,
        process_group: object | None = None,
        torch: Any | None = None,
        dist: Any | None = None,
    ) -> None:
        if not isinstance(dtype, DataType):
            raise TypeError("dtype must be a DataType")
        if not isinstance(wire, QuantizedWire):
            raise TypeError("wire must be a QuantizedWire")
        if layout_generation < 0:
            raise ValueError("layout_generation must be non-negative")
        if isinstance(max_numel, bool) or not isinstance(max_numel, int):
            raise TypeError("max_numel must be an integer")
        if max_numel <= 0:
            raise ValueError("max_numel must be positive")
        self.dtype = dtype
        self.wire = wire
        self.layout_generation = layout_generation
        self.max_numel = max_numel
        self._status = extension_status
        self._group = process_group
        self._torch = torch or import_module("torch")
        self._dist = dist or import_module("torch.distributed")
        self._host_metadata: Any | None = None
        self._metadata_ready: Any | None = None

    def run(self, tensor: Any) -> CompletionWork[tuple[Any, ...]]:
        torch = self._torch
        dist = self._dist
        world_size = dist.get_world_size(self._group)
        original_numel = int(tensor.numel())
        if original_numel > self.max_numel:
            raise ValueError(
                f"dynamic tensor numel {original_numel} exceeds max_numel "
                f"{self.max_numel}"
            )
        prepared = _pad(tensor, self.wire.group_size)
        valid_payload = payload_nbytes(
            original_numel,
            dtype=self.dtype,
            wire=self.wire,
        )
        payload = torch.empty(valid_payload, dtype=torch.uint8, device=tensor.device)
        quantize_into(
            prepared,
            payload,
            self.wire,
            extension_status=self._status,  # type: ignore[arg-type]
        )
        packet = MetadataPacket(
            shape=tuple(tensor.shape),
            dtype=self.dtype,
            wire=self.wire,
            payload_numel=valid_payload,
            layout_generation=self.layout_generation,
        )
        metadata = torch.tensor(
            packet.to_values(),
            dtype=torch.int64,
            device=tensor.device,
        )
        gathered_metadata = torch.empty(
            world_size * METADATA_PACKET_WORDS,
            dtype=torch.int64,
            device=tensor.device,
        )
        metadata_handle = dist.all_gather_into_tensor(
            gathered_metadata,
            metadata,
            group=self._group,
            async_op=True,
        )
        # CUDA Work.wait establishes current-stream ordering without copying the
        # fixed packet through Python object serialization.
        metadata_handle.wait()
        host_metadata = self._host_metadata_buffer(world_size)
        host_metadata.copy_(
            gathered_metadata.reshape(world_size, -1),
            non_blocking=True,
        )
        metadata_ready = self._metadata_event()
        metadata_ready.record(torch.cuda.current_stream(tensor.device))

        max_payload = payload_nbytes(
            self.max_numel,
            dtype=self.dtype,
            wire=self.wire,
        )
        stride = aligned_payload_stride((max_payload,))
        send = torch.zeros(stride, dtype=torch.uint8, device=tensor.device)
        send[:valid_payload].copy_(payload)
        gathered_payload = torch.empty(
            world_size * stride,
            dtype=torch.uint8,
            device=tensor.device,
        )
        payload_handle = dist.all_gather_into_tensor(
            gathered_payload,
            send,
            group=self._group,
            async_op=True,
        )
        metadata_ready.synchronize()
        packets = _decode_metadata_packets(host_metadata, world_size)
        for received in packets:
            if received.dtype is not self.dtype or received.wire != self.wire:
                raise ValueError("dynamic all-gather metadata schema mismatch")
            if received.layout_generation != self.layout_generation:
                raise ValueError("dynamic all-gather layout generation mismatch")
            if received.logical_numel > self.max_numel:
                raise ValueError("received dynamic shape exceeds max_numel")
            if received.payload_numel > stride:
                raise ValueError("received payload exceeds bounded rank stride")
        payload_handle.wait()
        outputs = []
        for rank, active in enumerate(packets):
            padded_numel = _aligned(active.logical_numel, active.wire.group_size)
            output = torch.empty(
                padded_numel,
                dtype=_torch_dtype(torch, active.dtype),
                device=tensor.device,
            )
            dequantize_into(
                gathered_payload.narrow(0, rank * stride, active.payload_numel),
                output,
                active.wire,
                dtype=active.dtype,
                extension_status=self._status,  # type: ignore[arg-type]
            )
            outputs.append(output[: active.logical_numel].reshape(active.shape))
        return CompletionWork(
            tuple(outputs),
            event=ImmediateCompletionEvent(),
            resources=(
                prepared,
                payload,
                metadata,
                gathered_metadata,
                host_metadata,
                metadata_handle,
                send,
                gathered_payload,
                payload_handle,
            ),
        )

    def _host_metadata_buffer(self, world_size: int) -> Any:
        torch = self._torch
        expected_shape = (world_size, METADATA_PACKET_WORDS)
        if self._host_metadata is None or tuple(self._host_metadata.shape) != expected_shape:
            self._host_metadata = torch.empty(
                expected_shape,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
        return self._host_metadata

    def _metadata_event(self) -> Any:
        if self._metadata_ready is None:
            self._metadata_ready = self._torch.cuda.Event()
        return self._metadata_ready


def _decode_metadata_packets(host_metadata: Any, world_size: int) -> tuple[MetadataPacket, ...]:
    rows = host_metadata.numpy()
    return tuple(
        MetadataPacket.from_values(tuple(map(int, rows[rank])))
        for rank in range(world_size)
    )


def _pad(tensor: Any, group_size: int) -> Any:
    flat = tensor.reshape(-1)
    if int(flat.numel()) % group_size == 0:
        return flat
    result = flat.new_zeros((_aligned(int(flat.numel()), group_size),))
    result[: flat.numel()].copy_(flat)
    return result


def _aligned(numel: int, group_size: int) -> int:
    return ((numel + group_size - 1) // group_size) * group_size


def aligned_payload_stride(payload_sizes: tuple[int, ...]) -> int:
    """Return a 16-byte aligned rank stride required by native vector loads."""

    if not payload_sizes or any(size < 0 for size in payload_sizes):
        raise ValueError("payload_sizes must be non-empty and non-negative")
    return _aligned(max(payload_sizes), 16)


def _torch_dtype(torch: Any, dtype: DataType) -> object:
    return {
        DataType.FP16: torch.float16,
        DataType.BF16: torch.bfloat16,
        DataType.FP32: torch.float32,
    }[dtype]
