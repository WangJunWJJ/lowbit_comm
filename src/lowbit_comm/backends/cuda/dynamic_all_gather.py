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
        self.dtype = dtype
        self.wire = wire
        self.layout_generation = layout_generation
        self._status = extension_status
        self._group = process_group
        self._torch = torch or import_module("torch")
        self._dist = dist or import_module("torch.distributed")

    def run(self, tensor: Any) -> CompletionWork[tuple[Any, ...]]:
        torch = self._torch
        dist = self._dist
        world_size = dist.get_world_size(self._group)
        original_numel = int(tensor.numel())
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
        dist.all_gather_into_tensor(
            gathered_metadata,
            metadata,
            group=self._group,
        )
        packets = tuple(
            MetadataPacket.from_values(values)
            for values in gathered_metadata.reshape(world_size, -1).tolist()
        )
        for received in packets:
            if received.dtype is not self.dtype or received.wire != self.wire:
                raise ValueError("dynamic all-gather metadata schema mismatch")
            if received.layout_generation != self.layout_generation:
                raise ValueError("dynamic all-gather layout generation mismatch")
        stride = aligned_payload_stride(
            tuple(active.payload_numel for active in packets)
        )
        send = torch.zeros(stride, dtype=torch.uint8, device=tensor.device)
        send[:valid_payload].copy_(payload)
        gathered_payload = torch.empty(
            world_size * stride,
            dtype=torch.uint8,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(
            gathered_payload,
            send,
            group=self._group,
        )
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
            resources=(prepared, payload, metadata, gathered_metadata, send, gathered_payload),
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
