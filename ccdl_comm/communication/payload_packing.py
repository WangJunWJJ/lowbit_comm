from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.gather_reduce import GatheredPayloads


DEFAULT_FUSED_PAYLOAD_MIN_NUMEL = 4_000_000


@dataclass(frozen=True)
class TensorFieldSchema:
    """Byte layout for one tensor field inside a packed payload."""

    name: str
    dtype: Any
    shape: tuple[int, ...]
    numel: int
    byte_count: int
    offset: int


@dataclass(frozen=True)
class FusedPayloadPacker:
    """Pack a compressed payload and tensor metadata into one byte tensor."""

    buffer_schema: TensorFieldSchema
    metadata_schemas: tuple[TensorFieldSchema, ...]
    scalar_metadata: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: CompressedPayload) -> "FusedPayloadPacker":
        tensor_metadata = {
            key: value
            for key, value in dict(payload.metadata).items()
            if _is_packable_tensor(value)
        }
        if not tensor_metadata:
            raise ValueError("fused payload all-gather requires tensor metadata")

        offset = 0
        buffer_schema = _make_schema("buffer", payload.buffer, offset)
        offset += buffer_schema.byte_count
        metadata_schemas: list[TensorFieldSchema] = []
        for key in sorted(tensor_metadata):
            schema = _make_schema(key, tensor_metadata[key], offset)
            metadata_schemas.append(schema)
            offset += schema.byte_count
        scalar_metadata = {
            key: value
            for key, value in dict(payload.metadata).items()
            if key not in tensor_metadata
        }
        return cls(
            buffer_schema=buffer_schema,
            metadata_schemas=tuple(metadata_schemas),
            scalar_metadata=scalar_metadata,
        )

    def pack(self, payload: CompressedPayload) -> Any:
        chunks = [_as_uint8(payload.buffer)]
        metadata = dict(payload.metadata)
        for schema in self.metadata_schemas:
            chunks.append(_as_uint8(metadata[schema.name]))
        return _torch_cat(chunks)

    def unpack(self, packed: Any, *, shape: tuple[int, ...], dtype: str) -> CompressedPayload:
        buffer = _unpack_tensor(packed, self.buffer_schema)
        metadata = dict(self.scalar_metadata)
        for schema in self.metadata_schemas:
            metadata[schema.name] = _unpack_tensor(packed, schema)
        return CompressedPayload(buffer=buffer, shape=shape, dtype=dtype, metadata=metadata)


@dataclass(frozen=True)
class FusedPayloadAllGather:
    """Callable fused all-gather with exposed packer for tests and adapters."""

    byte_all_gather: Callable[[Any], GatheredPayloads]
    packer: FusedPayloadPacker
    shape: tuple[int, ...]
    dtype: str

    def __call__(self, payload: CompressedPayload) -> GatheredPayloads:
        packed = self.packer.pack(payload)
        gathered = self.byte_all_gather(packed)
        return GatheredPayloads(
            payloads=[
                self.packer.unpack(remote_packed, shape=self.shape, dtype=self.dtype)
                for remote_packed in gathered.payloads
            ],
            world_size=gathered.world_size,
        )


def make_fused_payload_all_gather(
    byte_all_gather: Callable[[Any], GatheredPayloads],
) -> Callable[[CompressedPayload], GatheredPayloads]:
    """Create an all-gather adapter that fuses payload buffer and tensor metadata."""

    holder: dict[str, FusedPayloadPacker] = {}

    def fused_payload_all_gather(payload: CompressedPayload) -> GatheredPayloads:
        packer = holder.get("packer")
        if packer is None:
            packer = FusedPayloadPacker.from_payload(payload)
            holder["packer"] = packer
        adapter = FusedPayloadAllGather(
            byte_all_gather=byte_all_gather,
            packer=packer,
            shape=payload.shape,
            dtype=payload.dtype,
        )
        return adapter(payload)

    fused_payload_all_gather.packer = _LazyPacker(holder, byte_all_gather)  # type: ignore[attr-defined]
    return fused_payload_all_gather


def make_payload_all_gather(
    buffer_all_gather: Callable[[Any], GatheredPayloads],
) -> Callable[[CompressedPayload], GatheredPayloads]:
    """Create an all-gather adapter that gathers payload buffers and metadata."""

    def payload_all_gather(payload: CompressedPayload) -> GatheredPayloads:
        gathered = buffer_all_gather(payload.buffer)
        gathered_metadata = _gather_payload_metadata(payload.metadata, buffer_all_gather, gathered.world_size)
        return GatheredPayloads(
            payloads=[
                CompressedPayload(
                    buffer=buffer,
                    shape=payload.shape,
                    dtype=payload.dtype,
                    metadata={key: values[index] for key, values in gathered_metadata.items()},
                )
                for index, buffer in enumerate(gathered.payloads)
            ],
            world_size=gathered.world_size,
        )

    return payload_all_gather


def should_fuse_payload(
    tensor: Any,
    *,
    enabled: bool,
    min_numel: int = DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
) -> bool:
    """Return whether fused payload packing should be used for a tensor."""

    if not enabled:
        return False
    if min_numel <= 0:
        return True
    numel = _numel(tensor)
    return numel is not None and numel >= min_numel


class _LazyPacker:
    def __init__(self, holder: dict[str, FusedPayloadPacker], byte_all_gather: Callable[[Any], GatheredPayloads]) -> None:
        self._holder = holder
        self._byte_all_gather = byte_all_gather

    def pack(self, payload: CompressedPayload) -> Any:
        packer = self._holder.get("packer")
        if packer is None:
            packer = FusedPayloadPacker.from_payload(payload)
            self._holder["packer"] = packer
        return packer.pack(payload)


def _make_schema(name: str, tensor: Any, offset: int) -> TensorFieldSchema:
    contiguous = tensor.contiguous() if callable(getattr(tensor, "contiguous", None)) else tensor
    element_size = contiguous.element_size()
    numel = contiguous.numel()
    return TensorFieldSchema(
        name=name,
        dtype=contiguous.dtype,
        shape=tuple(contiguous.shape),
        numel=int(numel),
        byte_count=int(numel * element_size),
        offset=offset,
    )


def _as_uint8(tensor: Any) -> Any:
    torch = _import_torch()
    contiguous = tensor.contiguous() if callable(getattr(tensor, "contiguous", None)) else tensor
    return contiguous.view(torch.uint8).reshape(-1)


def _unpack_tensor(packed: Any, schema: TensorFieldSchema) -> Any:
    byte_view = packed.narrow(0, schema.offset, schema.byte_count)
    return byte_view.view(schema.dtype).reshape(schema.shape)


def _torch_cat(chunks: list[Any]) -> Any:
    torch = _import_torch()
    return torch.cat(chunks, dim=0)


def _import_torch():
    import torch

    return torch


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _is_packable_tensor(value: Any) -> bool:
    return _is_tensor_like(value) and hasattr(value, "numel")


def _gather_payload_metadata(
    metadata: dict[str, Any] | Any,
    tensor_all_gather: Callable[[Any], GatheredPayloads],
    world_size: int,
) -> dict[str, list[Any]]:
    gathered: dict[str, list[Any]] = {}
    for key, value in dict(metadata).items():
        if _is_tensor_like(value):
            gathered[key] = list(tensor_all_gather(value).payloads)
    for key, value in dict(metadata).items():
        if key not in gathered:
            gathered[key] = [value for _ in range(world_size)]
    return gathered


def _numel(tensor: Any) -> int | None:
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        return int(numel())
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
