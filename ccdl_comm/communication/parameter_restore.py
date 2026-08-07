"""Compressed restoration of updated rank-local parameter shards."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus, load_cuda_extension
from ccdl_comm.optim import UpdatedParameterShard
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    inplace_dequantize_gathered,
    quantize_tensor,
)
from ccdl_comm.work import CollectiveWork


@dataclass(slots=True)
class _RestoreWorkspace:
    send: Any
    gathered: Any
    payload_numel: int
    in_flight: bool = False


class _ReleaseOnErrorHandle:
    def __init__(self, handle: Any, workspace: _RestoreWorkspace) -> None:
        self._handle = handle
        self._workspace = workspace

    def wait(self) -> Any:
        wait = getattr(self._handle, "wait", None)
        try:
            return wait() if callable(wait) else None
        except BaseException:
            self._workspace.in_flight = False
            raise

    def is_completed(self) -> bool:
        for name in ("is_completed", "query"):
            query = getattr(self._handle, name, None)
            if callable(query):
                return bool(query())
        return False

    def get_future(self) -> Any | None:
        get_future = getattr(self._handle, "get_future", None)
        return get_future() if callable(get_future) else None


class TorchCompressedParameterRestore:
    """All-gather INT8 parameter shards and decode directly into flat storage."""

    def __init__(
        self,
        *,
        config: CompressionConfig,
        dtype: str,
        import_module: Callable[[str], Any] = import_module,
        quantize: Callable[..., Any] | None = None,
        dequantize_gathered: Callable[..., bool] | None = None,
        quantized_allocator: Callable[..., Any] | None = None,
        supports_compressed: Callable[[UpdatedParameterShard, Any, int], bool] | None = None,
        completion_manager: CudaCompletionManager | None = None,
        extension_status: CudaExtensionStatus | None = None,
    ) -> None:
        if not isinstance(config, CompressionConfig):
            raise TypeError("config must be a CompressionConfig")
        if not isinstance(dtype, str) or not dtype.strip():
            raise TypeError("dtype must be a non-empty string")
        self.config = config
        self._dtype = dtype
        self._dist = import_module("torch.distributed")
        self._extension_status = extension_status or load_cuda_extension()
        self._quantize = quantize or self._default_quantize
        self._dequantize_gathered = (
            dequantize_gathered or self._default_dequantize_gathered
        )
        self._quantized_allocator = quantized_allocator or allocate_quantized_buffer
        self._supports_compressed = supports_compressed or self._default_supports
        self._completion_manager = completion_manager or CudaCompletionManager(
            extension_status=self._extension_status
        )
        self._workspaces: dict[tuple[Any, ...], _RestoreWorkspace] = {}

    def restore(
        self,
        updated: UpdatedParameterShard,
        *,
        out: Any,
        async_op: bool = True,
    ) -> CollectiveWork[Any]:
        """Restore one replicated padded parameter buffer in collective order."""

        self._validate(updated, out)
        workspace = self._workspace_for(updated)
        if not self._supports_compressed(updated, out, workspace.payload_numel):
            handle = self._dist.all_gather_into_tensor(
                out,
                updated.shard,
                async_op=async_op,
            )
            return self._completion_manager.create_work(
                result=out,
                handle=handle,
                complete=lambda: out,
                resources=(updated.shard, out),
            )
        if workspace.in_flight:
            raise RuntimeError("parameter restore workspace is in flight")

        workspace.in_flight = True
        try:
            result = self._quantize(
                updated.shard,
                self.config,
                output=workspace.send,
            )
            if result is not workspace.send:
                raise RuntimeError("quantize must return the caller-owned send workspace")
            handle = self._dist.all_gather_into_tensor(
                workspace.gathered,
                workspace.send,
                async_op=async_op,
            )
        except BaseException:
            workspace.in_flight = False
            raise

        if handle is not None:
            handle = _ReleaseOnErrorHandle(handle, workspace)
        return self._completion_manager.create_work(
            result=out,
            handle=handle,
            complete=lambda: self._finish_int8_restore(updated, workspace, out),
            resources=(workspace.send, workspace.gathered, out),
        )

    def _workspace_for(self, updated: UpdatedParameterShard) -> _RestoreWorkspace:
        key = (
            str(getattr(updated.shard, "device", "")),
            self._dtype,
            updated.shard_numel,
            updated.world_size,
            self.config.bit,
            self.config.group_size,
            self.config.topk,
            self.config.quant_type,
            self.config.compact,
        )
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace
        send = self._quantized_allocator(
            updated.shard,
            self.config,
            dtype=self._dtype,
        )
        payload_numel = _tensor_numel(send, "quantized send workspace")
        gathered = send.new_empty((payload_numel * updated.world_size,))
        workspace = _RestoreWorkspace(
            send=send,
            gathered=gathered,
            payload_numel=payload_numel,
        )
        self._workspaces[key] = workspace
        return workspace

    def _finish_int8_restore(
        self,
        updated: UpdatedParameterShard,
        workspace: _RestoreWorkspace,
        out: Any,
    ) -> Any:
        try:
            supported = self._dequantize_gathered(
                workspace.gathered,
                out,
                self.config,
                dtype=self._dtype,
                world_size=updated.world_size,
                payload_numel=workspace.payload_numel,
                payload_stride=workspace.payload_numel,
                shard_numel=updated.shard_numel,
            )
            if not supported:
                raise RuntimeError(
                    "gathered dequantize rejected after INT8 collective"
                )
            return out
        finally:
            workspace.in_flight = False

    def _validate(self, updated: UpdatedParameterShard, out: Any) -> None:
        if not isinstance(updated, UpdatedParameterShard):
            raise TypeError("updated must be an UpdatedParameterShard")
        if _tensor_numel(out, "output") != updated.padded_numel:
            raise ValueError("output numel must equal updated padded_numel")
        _require_contiguous(out, "output")
        _require_matching_property(updated.shard, out, "dtype")
        _require_matching_property(updated.shard, out, "device")
        get_world_size = getattr(self._dist, "get_world_size", None)
        if callable(get_world_size) and int(get_world_size()) != updated.world_size:
            raise ValueError("distributed world size does not match updated shard")

    def _default_supports(
        self,
        updated: UpdatedParameterShard,
        out: Any,
        payload_numel: int,
    ) -> bool:
        del out
        module = self._extension_status.module
        return bool(
            self._extension_status.available
            and module is not None
            and callable(getattr(module, "inplace_dequantize_gathered", None))
            and self.config.bit == 8
            and self.config.group_size == 64
            and self.config.topk == 0
            and self.config.quant_type == "linear"
            and self._dtype in {"fp16", "bf16", "fp32"}
            and 1 <= updated.world_size <= 8
            and updated.shard_numel > 0
            and payload_numel % 16 == 0
        )

    def _default_quantize(self, tensor: Any, config: CompressionConfig, *, output: Any) -> Any:
        return quantize_tensor(
            tensor,
            config,
            output=output,
            extension_status=self._extension_status,
        )

    def _default_dequantize_gathered(self, *args: Any, **kwargs: Any) -> bool:
        return inplace_dequantize_gathered(
            *args,
            **kwargs,
            extension_status=self._extension_status,
        )


def _tensor_numel(tensor: Any, name: str) -> int:
    numel = getattr(tensor, "numel", None)
    if not callable(numel):
        raise TypeError(f"{name} must expose numel()")
    return int(numel())


def _require_contiguous(tensor: Any, name: str) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(is_contiguous) or not bool(is_contiguous()):
        raise ValueError(f"{name} must be contiguous")


def _require_matching_property(left: Any, right: Any, name: str) -> None:
    if getattr(left, name, None) != getattr(right, name, None):
        raise ValueError(f"output {name} must match updated shard {name}")


__all__ = ["TorchCompressedParameterRestore"]
