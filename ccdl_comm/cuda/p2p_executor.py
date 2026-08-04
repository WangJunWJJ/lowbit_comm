"""Compiled CUDA point-to-point executors with immutable transport bindings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError
from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    dequantize_tensor,
    quantize_tensor,
)
from ccdl_comm.quantization.sizing import estimate_quantized_size
from ccdl_comm.work import CollectiveWork, bind_execution_work


P2P_METADATA_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class P2PMetadata:
    """Static payload contract retained by every in-flight P2P work."""

    protocol_version: int
    shape: tuple[int, ...]
    dtype: str
    payload_numel: int


class CudaP2PExecutor:
    """Execute quantized send or receive with compile-time transport metadata."""

    def __init__(
        self,
        *,
        direction: str,
        peer: int | None,
        tensor: Any,
        config: CompressionConfig,
        group: Any | None = None,
        tag: int = 0,
        dtype: str = "auto",
        extension_status: Any | None = None,
        import_module_fn: Callable[[str], Any] = import_module,
        quantize: Callable[..., Any] = quantize_tensor,
        allocate_quantized: Callable[..., Any] = allocate_quantized_buffer,
        dequantize: Callable[..., Any] = dequantize_tensor,
        completion_manager: CudaCompletionManager | Any | None = None,
    ) -> None:
        if direction not in {"send", "recv"}:
            raise ValueError("direction must be 'send' or 'recv'")
        if direction == "send" and peer is None:
            raise ValueError("compiled send requires a destination peer")
        if peer is not None and (
            isinstance(peer, bool) or not isinstance(peer, int) or peer < 0
        ):
            raise ValueError("peer must be a non-negative integer or None")
        if isinstance(tag, bool) or not isinstance(tag, int) or tag < 0:
            raise ValueError("tag must be a non-negative integer")
        if not isinstance(config, CompressionConfig):
            raise TypeError("config must be a CompressionConfig")

        self._direction = direction
        self._peer = peer
        self._group = group
        self._tag = tag
        self._config = config
        self._dtype = _resolve_dtype(dtype, tensor)
        self._shape = tuple(tensor.shape)
        self._device = str(getattr(tensor, "device", "cuda"))
        self._dist = _distributed(import_module_fn)
        self._extension_status = extension_status
        self._quantize = quantize
        self._allocate_quantized = allocate_quantized
        self._dequantize = dequantize
        self._completion_manager = completion_manager or CudaCompletionManager(
            extension_status=extension_status
        )
        estimate = estimate_quantized_size(
            _numel(self._shape),
            dtype=self._dtype,
            config=config,
        )
        self._metadata = P2PMetadata(
            protocol_version=P2P_METADATA_PROTOCOL_VERSION,
            shape=self._shape,
            dtype=self._dtype,
            payload_numel=estimate.quantized_bytes,
        )
        self.execution_info = ExecutionInfo(
            requested_strategy="compressed_p2p",
            executed_strategy="compressed_p2p",
            backend="cuda",
            fallback_used=False,
            fallback_reason=None,
            stage_names=(),
            original_bytes=estimate.original_bytes,
            compressed_bytes=estimate.quantized_bytes,
            compression_ratio=estimate.compression_ratio or 1.0,
            workspace_cache_hit=False,
            async_capable=True,
            fast_path=f"cuda_compiled_{direction}",
            details={
                "direction": direction,
                "peer": peer,
                "tag": tag,
                "dtype": self._dtype,
                "shape": self._shape,
                "metadata_protocol_version": P2P_METADATA_PROTOCOL_VERSION,
            },
        )
        self.execution_counters = ExecutionCounters()

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def peer(self) -> int | None:
        return self._peer

    @property
    def group(self) -> Any | None:
        return self._group

    @property
    def tag(self) -> int:
        return self._tag

    @property
    def config(self) -> CompressionConfig:
        return self._config

    @property
    def metadata(self) -> P2PMetadata:
        return self._metadata

    def run(self, tensor: object) -> CollectiveWork[object]:
        """Submit one operation without resolving transport policy again."""

        self.execution_counters._record_run()
        try:
            _validate_tensor(
                tensor,
                shape=self._shape,
                dtype=self._dtype,
                device=self._device,
            )
            work = self._send(tensor) if self._direction == "send" else self._recv(tensor)
            return bind_execution_work(
                work,
                self.execution_info,
                self.execution_counters,
            )
        except BaseException:
            self.execution_counters._record_failed()
            raise

    def _send(self, tensor: object) -> object:
        payload = self._quantize(
            tensor,
            self._config,
            extension_status=self._extension_status,
        )
        _validate_payload(payload, self._metadata.payload_numel)
        handle = self._dist.isend(
            payload,
            self._peer,
            group=self._group,
            tag=self._tag,
        )
        return self._completion_manager.create_work(
            result=None,
            handle=handle,
            resources=(tensor, payload, self._metadata),
        )

    def _recv(self, tensor: object) -> object:
        payload = self._allocate_quantized(
            tensor,
            self._config,
            dtype=self._dtype,
        )
        _validate_payload(payload, self._metadata.payload_numel)
        handle = self._dist.irecv(
            payload,
            self._peer,
            group=self._group,
            tag=self._tag,
        )

        def complete() -> object:
            self._dequantize(
                payload,
                self._shape,
                self._config,
                dtype=self._dtype,
                extension_status=self._extension_status,
                output=tensor,
                reduce_op="none",
            )
            return tensor

        return self._completion_manager.create_work(
            result=tensor,
            handle=handle,
            complete=complete,
            resources=(tensor, payload, self._metadata),
        )


def compile_p2p_executor(
    *,
    direction: str,
    peer: int | None,
    tensor: Any,
    config: CompressionConfig,
    **kwargs: Any,
) -> CudaP2PExecutor:
    """Compile one immutable quantized P2P endpoint."""

    return CudaP2PExecutor(
        direction=direction,
        peer=peer,
        tensor=tensor,
        config=config,
        **kwargs,
    )


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


def _validate_tensor(
    tensor: object,
    *,
    shape: tuple[int, ...],
    dtype: str,
    device: str,
) -> None:
    if tuple(getattr(tensor, "shape", ())) != shape:
        raise ValueError(
            f"compiled P2P shape is {shape}; received {tuple(getattr(tensor, 'shape', ()))}"
        )
    active_dtype = _resolve_dtype("auto", tensor)
    if active_dtype != dtype:
        raise TypeError(
            f"compiled P2P dtype is {dtype!r}; received {active_dtype!r}"
        )
    active_device = str(getattr(tensor, "device", device))
    if active_device != device:
        raise ValueError(
            f"compiled P2P device is {device!r}; received {active_device!r}"
        )


def _validate_payload(payload: object, expected_numel: int) -> None:
    actual_numel = int(payload.numel())
    if actual_numel != expected_numel:
        raise ValueError(
            f"compiled P2P payload requires {expected_numel} bytes; "
            f"received {actual_numel}"
        )


def _resolve_dtype(dtype: str, tensor: Any) -> str:
    if dtype != "auto":
        return dtype
    name = str(getattr(tensor, "dtype", "")).lower()
    if "bfloat16" in name or "bf16" in name:
        return "bf16"
    if "float32" in name or "fp32" in name:
        return "fp32"
    if "float16" in name or "fp16" in name or name.endswith("half"):
        return "fp16"
    raise ValueError(f"cannot infer CCDL dtype from tensor dtype: {name!r}")


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)
