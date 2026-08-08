"""Quantized weight-difference restore and full-precision refresh."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.parameter_delta import ParameterDeltaShard
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
class _ParameterDeltaWorkspace:
    send: Any
    gathered: Any
    decoded: Any
    refresh_send: Any
    refresh_gathered: Any
    payload_numel: int
    in_flight: bool = False


class _ReleaseWorkspaceOnError:
    def __init__(self, handle: Any, workspace: _ParameterDeltaWorkspace) -> None:
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


class TorchQuantizedParameterDeltaRestore:
    """All-gather qWD payloads and add them to a replicated model copy."""

    def __init__(
        self,
        *,
        config: CompressionConfig,
        model_dtype: str,
        import_module: Callable[[str], Any] = import_module,
        quantize: Callable[..., Any] | None = None,
        dequantize_add: Callable[..., bool] | None = None,
        overwrite: Callable[[Any, Any], Any] | None = None,
        quantized_allocator: Callable[..., Any] | None = None,
        supports_qwd: Callable[[Any, Any, int], bool] | None = None,
        completion_manager: CudaCompletionManager | None = None,
        extension_status: CudaExtensionStatus | None = None,
    ) -> None:
        if not isinstance(config, CompressionConfig):
            raise TypeError("config must be a CompressionConfig")
        if model_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("model_dtype must be fp16, bf16, or fp32")
        self.config = config
        self._model_dtype = model_dtype
        self._dist = import_module("torch.distributed")
        self._extension_status = extension_status or load_cuda_extension()
        self._quantize = quantize or self._default_quantize
        self._dequantize_add = dequantize_add or self._default_dequantize_add
        self._overwrite = overwrite or self._default_overwrite
        self._quantized_allocator = quantized_allocator or allocate_quantized_buffer
        self._supports_qwd = supports_qwd or self._default_supports_qwd
        self._completion_manager = completion_manager or CudaCompletionManager(
            extension_status=self._extension_status
        )
        self._workspaces: dict[tuple[Any, ...], _ParameterDeltaWorkspace] = {}
        self._last_fast_path: str | None = None
        self._last_fallback_reason: str | None = None

    @property
    def last_fast_path(self) -> str | None:
        return self._last_fast_path

    @property
    def last_fallback_reason(self) -> str | None:
        return self._last_fallback_reason

    def supports_qwd(self, updated: UpdatedParameterShard, out: Any) -> bool:
        """Return whether qWD can start without requiring a later fallback."""

        self._validate_master(updated, out)
        workspace = self._workspace_for(updated, out)
        supported = bool(
            self._supports_qwd(updated, out, workspace.payload_numel)
        )
        self._last_fallback_reason = (
            None if supported else "INT8 qWD capability is unavailable"
        )
        return supported

    def restore_delta(
        self,
        delta: ParameterDeltaShard,
        *,
        out: Any,
        async_op: bool = True,
    ) -> CollectiveWork[Any]:
        """Add one all-gathered quantized weight difference to ``out``."""

        self._validate_delta(delta, out)
        workspace = self._workspace_for(delta, out)
        if not self._supports_qwd(delta, out, workspace.payload_numel):
            self._last_fast_path = None
            self._last_fallback_reason = "INT8 qWD capability is unavailable"
            raise RuntimeError(
                "qWD restore is unsupported; select fp refresh before collective"
            )
        self._acquire(workspace)
        self._last_fast_path = "int8_qwd"
        self._last_fallback_reason = None
        try:
            packed = self._quantize(
                delta.shard,
                self.config,
                output=workspace.send,
            )
            if packed is not workspace.send:
                raise RuntimeError("quantize must return caller-owned qWD send workspace")
            handle = self._dist.all_gather_into_tensor(
                workspace.gathered,
                workspace.send,
                async_op=async_op,
            )
        except BaseException:
            workspace.in_flight = False
            raise
        handle = self._guard_handle(handle, workspace)
        return self._completion_manager.create_work(
            result=out,
            handle=handle,
            complete=lambda: self._finish_delta(delta, workspace, out),
            resources=(
                delta.shard,
                workspace.send,
                workspace.gathered,
                workspace.decoded,
                out,
            ),
        )

    def refresh(
        self,
        master: UpdatedParameterShard,
        *,
        out: Any,
        async_op: bool = True,
    ) -> CollectiveWork[Any]:
        """All-gather FP32 master shards and overwrite the model copy."""

        self._validate_master(master, out)
        workspace = self._workspace_for(master, out)
        self._acquire(workspace)
        self._last_fast_path = "fp_parameter_refresh"
        self._last_fallback_reason = None
        try:
            copied = workspace.refresh_send.copy_(master.shard)
            if copied is not workspace.refresh_send:
                raise RuntimeError(
                    "refresh copy must return caller-owned send workspace"
                )
            handle = self._dist.all_gather_into_tensor(
                workspace.refresh_gathered,
                workspace.refresh_send,
                async_op=async_op,
            )
        except BaseException:
            workspace.in_flight = False
            raise
        handle = self._guard_handle(handle, workspace)
        return self._completion_manager.create_work(
            result=out,
            handle=handle,
            complete=lambda: self._finish_refresh(workspace, out),
            resources=(
                master.shard,
                workspace.refresh_send,
                workspace.refresh_gathered,
                out,
            ),
        )

    def workspace_pointers(self) -> dict[str, tuple[int, ...]]:
        """Return stable workspace identities for allocation gates."""

        return {
            name: tuple(
                _tensor_pointer(getattr(workspace, name))
                for workspace in self._workspaces.values()
            )
            for name in (
                "send",
                "gathered",
                "decoded",
                "refresh_send",
                "refresh_gathered",
            )
        }

    def _workspace_for(
        self,
        shard_metadata: ParameterDeltaShard | UpdatedParameterShard,
        out: Any,
    ) -> _ParameterDeltaWorkspace:
        shard = shard_metadata.shard
        key = (
            str(getattr(shard, "device", "")),
            self._model_dtype,
            shard_metadata.shard_numel,
            shard_metadata.padded_numel,
            shard_metadata.world_size,
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
            shard,
            self.config,
            dtype="fp32",
        )
        payload_numel = _tensor_numel(send, "quantized qWD send workspace")
        gathered = send.new_empty((payload_numel * shard_metadata.world_size,))
        decoded = out.new_empty((shard_metadata.padded_numel,))
        refresh_send = shard.new_empty((shard_metadata.shard_numel,))
        refresh_gathered = shard.new_empty((shard_metadata.padded_numel,))
        workspace = _ParameterDeltaWorkspace(
            send=send,
            gathered=gathered,
            decoded=decoded,
            refresh_send=refresh_send,
            refresh_gathered=refresh_gathered,
            payload_numel=payload_numel,
        )
        self._workspaces[key] = workspace
        return workspace

    def _finish_delta(
        self,
        delta: ParameterDeltaShard,
        workspace: _ParameterDeltaWorkspace,
        out: Any,
    ) -> Any:
        try:
            supported = self._dequantize_add(
                workspace.gathered,
                out,
                workspace.decoded,
                self.config,
                dtype=self._model_dtype,
                world_size=delta.world_size,
                payload_numel=workspace.payload_numel,
                payload_stride=workspace.payload_numel,
                shard_numel=delta.shard_numel,
                original_numel=delta.original_numel,
            )
            if not supported:
                raise RuntimeError(
                    "qWD dequantize-add rejected after INT8 collective"
                )
            return out
        finally:
            workspace.in_flight = False

    def _finish_refresh(
        self,
        workspace: _ParameterDeltaWorkspace,
        out: Any,
    ) -> Any:
        try:
            result = self._overwrite(out, workspace.refresh_gathered)
            if result is not out:
                raise RuntimeError("refresh overwrite must return output")
            return out
        finally:
            workspace.in_flight = False

    def _validate_delta(self, delta: ParameterDeltaShard, out: Any) -> None:
        if not isinstance(delta, ParameterDeltaShard):
            raise TypeError("delta must be a ParameterDeltaShard")
        self._validate_common(delta, out)
        if _canonical_dtype(delta.shard) != "fp32":
            raise ValueError("qWD delta shard must use fp32")

    def _validate_master(self, master: UpdatedParameterShard, out: Any) -> None:
        if not isinstance(master, UpdatedParameterShard):
            raise TypeError("master must be an UpdatedParameterShard")
        self._validate_common(master, out)
        if _canonical_dtype(master.shard) != "fp32":
            raise ValueError("master shard must use fp32")

    def _validate_common(
        self,
        metadata: ParameterDeltaShard | UpdatedParameterShard,
        out: Any,
    ) -> None:
        if _tensor_numel(out, "model output") != metadata.padded_numel:
            raise ValueError("model output numel must equal padded_numel")
        _require_contiguous(out, "model output")
        _require_contiguous(metadata.shard, "parameter shard")
        if _canonical_dtype(out) != self._model_dtype:
            raise ValueError("model output dtype must match model_dtype")
        if getattr(metadata.shard, "device", None) != getattr(out, "device", None):
            raise ValueError("model output device must match parameter shard")
        get_world_size = getattr(self._dist, "get_world_size", None)
        if callable(get_world_size) and int(get_world_size()) != metadata.world_size:
            raise ValueError("distributed world size does not match parameter shard")

    @staticmethod
    def _acquire(workspace: _ParameterDeltaWorkspace) -> None:
        if workspace.in_flight:
            raise RuntimeError("qWD workspace is in flight")
        workspace.in_flight = True

    @staticmethod
    def _guard_handle(
        handle: Any,
        workspace: _ParameterDeltaWorkspace,
    ) -> Any:
        return (
            _ReleaseWorkspaceOnError(handle, workspace)
            if handle is not None
            else None
        )

    def _default_supports_qwd(
        self,
        metadata: ParameterDeltaShard | UpdatedParameterShard,
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
            and self._model_dtype in {"fp16", "bf16", "fp32"}
            and 1 <= metadata.world_size <= 8
            and metadata.shard_numel > 0
            and payload_numel % 16 == 0
        )

    def _default_quantize(
        self,
        tensor: Any,
        config: CompressionConfig,
        *,
        output: Any,
    ) -> Any:
        return quantize_tensor(
            tensor,
            config,
            output=output,
            extension_status=self._extension_status,
        )

    def _default_dequantize_add(
        self,
        gathered: Any,
        out: Any,
        decoded: Any,
        config: CompressionConfig,
        **kwargs: Any,
    ) -> bool:
        original_numel = kwargs.pop("original_numel")
        supported = inplace_dequantize_gathered(
            gathered,
            decoded,
            config,
            extension_status=self._extension_status,
            **kwargs,
        )
        if not supported:
            return False
        if original_numel < _tensor_numel(decoded, "decoded qWD workspace"):
            decoded.narrow(
                0,
                original_numel,
                _tensor_numel(decoded, "decoded qWD workspace") - original_numel,
            ).zero_()
        result = out.add_(decoded)
        if result is not out:
            raise RuntimeError("qWD add must update and return output")
        return True

    @staticmethod
    def _default_overwrite(out: Any, gathered: Any) -> Any:
        return out.copy_(gathered)


def _canonical_dtype(tensor: Any) -> str:
    value = str(getattr(tensor, "dtype", "")).lower()
    return {
        "float16": "fp16",
        "torch.float16": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "bf16": "bf16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "fp32": "fp32",
    }.get(value, value)


def _tensor_numel(tensor: Any, name: str) -> int:
    numel = getattr(tensor, "numel", None)
    if not callable(numel):
        raise TypeError(f"{name} must expose numel()")
    return int(numel())


def _tensor_pointer(tensor: Any) -> int:
    data_ptr = getattr(tensor, "data_ptr", None)
    return int(data_ptr()) if callable(data_ptr) else id(tensor)


def _require_contiguous(tensor: Any, name: str) -> None:
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if not callable(is_contiguous) or not bool(is_contiguous()):
        raise ValueError(f"{name} must be contiguous")


__all__ = ["TorchQuantizedParameterDeltaRestore"]
