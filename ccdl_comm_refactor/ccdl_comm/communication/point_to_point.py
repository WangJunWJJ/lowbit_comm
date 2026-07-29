from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError
from ccdl_comm.quantization.codec import allocate_quantized_buffer, dequantize_tensor, quantize_tensor


@dataclass
class PointToPointWork:
    """Work wrapper that can complete deferred receive-side dequantization."""

    handle: Any
    result: Any | None = None
    complete: Callable[[], Any] | None = None

    def wait(self) -> Any:
        wait = getattr(self.handle, "wait", None)
        if callable(wait):
            wait()
        if self.complete is not None:
            return self.complete()
        return self.result


def qsend(
    tensor: Any,
    dst: int,
    *,
    config: CompressionConfig,
    group: Any | None = None,
    tag: int = 0,
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
) -> None:
    """Quantize `tensor` and send the compressed buffer to `dst`."""

    dist = _distributed(import_module_fn)
    buffer = quantize(tensor, config, extension_status=extension_status)
    dist.send(buffer, dst, group=group, tag=tag)


def qrecv(
    tensor: Any,
    src: int | None = None,
    *,
    config: CompressionConfig,
    group: Any | None = None,
    tag: int = 0,
    dtype: str = "auto",
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    allocate_quantized: Callable[..., Any] = allocate_quantized_buffer,
    dequantize: Callable[..., Any] = dequantize_tensor,
) -> Any:
    """Receive a compressed buffer from `src` and dequantize it into `tensor`."""

    dist = _distributed(import_module_fn)
    active_dtype = _resolve_dtype(dtype, tensor)
    buffer = allocate_quantized(tensor, config, dtype=active_dtype)
    dist.recv(buffer, src, group=group, tag=tag)
    dequantize(
        buffer,
        tuple(tensor.shape),
        config,
        dtype=active_dtype,
        extension_status=extension_status,
        output=tensor,
        reduce_op="none",
    )
    return tensor


def iqsend(
    tensor: Any,
    dst: int,
    *,
    config: CompressionConfig,
    group: Any | None = None,
    tag: int = 0,
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
) -> Any:
    """Quantize `tensor` and start a non-blocking send."""

    dist = _distributed(import_module_fn)
    buffer = quantize(tensor, config, extension_status=extension_status)
    return dist.isend(buffer, dst, group=group, tag=tag)


def iqrecv(
    tensor: Any,
    src: int | None = None,
    *,
    config: CompressionConfig,
    group: Any | None = None,
    tag: int = 0,
    dtype: str = "auto",
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    allocate_quantized: Callable[..., Any] = allocate_quantized_buffer,
    dequantize: Callable[..., Any] = dequantize_tensor,
) -> PointToPointWork:
    """Start a non-blocking receive and dequantize into `tensor` on `wait()`."""

    dist = _distributed(import_module_fn)
    active_dtype = _resolve_dtype(dtype, tensor)
    buffer = allocate_quantized(tensor, config, dtype=active_dtype)
    handle = dist.irecv(buffer, src, group=group, tag=tag)

    def complete() -> Any:
        dequantize(
            buffer,
            tuple(tensor.shape),
            config,
            dtype=active_dtype,
            extension_status=extension_status,
            output=tensor,
            reduce_op="none",
        )
        return tensor

    return PointToPointWork(handle=handle, result=tensor, complete=complete)


def _distributed(import_module_fn: Callable[[str], Any]) -> Any:
    try:
        dist = import_module_fn("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError("torch.distributed is not available") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError("torch.distributed is not initialized")
    return dist


def _resolve_dtype(dtype: str, tensor: Any) -> str:
    if dtype != "auto":
        return dtype
    name = str(getattr(tensor, "dtype", "")).lower()
    if "bfloat16" in name or "bf16" in name:
        return "bf16"
    if "float32" in name or "fp32" in name:
        return "fp32"
    return "fp16"
