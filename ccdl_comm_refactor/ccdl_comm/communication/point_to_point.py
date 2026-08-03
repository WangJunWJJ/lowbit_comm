from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.collectives.work import CollectiveWork, CompletionWork
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError
from ccdl_comm.quantization.codec import allocate_quantized_buffer, dequantize_tensor, quantize_tensor


PointToPointWork = CompletionWork


def compile_qsend(
    tensor: Any,
    dst: int,
    *,
    config: CompressionConfig,
    **kwargs: Any,
) -> Any:
    """Compile a reusable quantized send endpoint for one tensor contract."""

    from ccdl_comm.cuda.p2p_executor import compile_p2p_executor

    return compile_p2p_executor(
        direction="send",
        peer=dst,
        tensor=tensor,
        config=config,
        **kwargs,
    )


def compile_qrecv(
    tensor: Any,
    src: int | None = None,
    *,
    config: CompressionConfig,
    **kwargs: Any,
) -> Any:
    """Compile a reusable quantized receive endpoint for one tensor contract."""

    from ccdl_comm.cuda.p2p_executor import compile_p2p_executor

    return compile_p2p_executor(
        direction="recv",
        peer=src,
        tensor=tensor,
        config=config,
        **kwargs,
    )


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
    completion_manager: CudaCompletionManager | Any | None = None,
    compiled_executor: Any | None = None,
) -> None:
    """Quantize `tensor` and send the compressed buffer to `dst`."""

    work = iqsend(
        tensor,
        dst,
        config=config,
        group=group,
        tag=tag,
        extension_status=extension_status,
        import_module_fn=import_module_fn,
        quantize=quantize,
        completion_manager=completion_manager,
        compiled_executor=compiled_executor,
    )
    work.wait()


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
    completion_manager: CudaCompletionManager | Any | None = None,
    compiled_executor: Any | None = None,
) -> Any:
    """Receive a compressed buffer from `src` and dequantize it into `tensor`."""

    active_dtype = (
        dtype
        if compiled_executor is not None
        else _resolve_dtype(dtype, tensor)
    )
    work = iqrecv(
        tensor,
        src,
        config=config,
        group=group,
        tag=tag,
        dtype=active_dtype,
        extension_status=extension_status,
        import_module_fn=import_module_fn,
        allocate_quantized=allocate_quantized,
        dequantize=dequantize,
        completion_manager=completion_manager,
        compiled_executor=compiled_executor,
    )
    return work.wait()


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
    completion_manager: CudaCompletionManager | Any | None = None,
    compiled_executor: Any | None = None,
) -> CollectiveWork[Any]:
    """Quantize `tensor` and start a non-blocking send."""

    if compiled_executor is not None:
        _validate_compiled_executor(
            compiled_executor,
            direction="send",
            peer=dst,
            config=config,
            group=group,
            tag=tag,
        )
        return compiled_executor.run(tensor)

    dist = _distributed(import_module_fn)
    buffer = quantize(tensor, config, extension_status=extension_status)
    handle = dist.isend(buffer, dst, group=group, tag=tag)
    manager = completion_manager or CudaCompletionManager()
    return manager.create_work(result=None, handle=handle, resources=(buffer,))


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
    completion_manager: CudaCompletionManager | Any | None = None,
    compiled_executor: Any | None = None,
) -> CollectiveWork[Any]:
    """Start a non-blocking receive and dequantize into `tensor` on `wait()`."""

    if compiled_executor is not None:
        _validate_compiled_executor(
            compiled_executor,
            direction="recv",
            peer=src,
            config=config,
            group=group,
            tag=tag,
        )
        return compiled_executor.run(tensor)

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

    manager = completion_manager or CudaCompletionManager()
    return manager.create_work(
        result=tensor,
        handle=handle,
        complete=complete,
        resources=(buffer, tensor),
    )


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


def _validate_compiled_executor(
    executor: Any,
    *,
    direction: str,
    peer: int | None,
    config: CompressionConfig,
    group: Any | None,
    tag: int,
) -> None:
    if getattr(executor, "direction", None) != direction:
        raise ValueError(
            f"compiled P2P executor direction must be {direction!r}"
        )
    if getattr(executor, "peer", None) != peer:
        raise ValueError(
            f"compiled P2P executor peer is {getattr(executor, 'peer', None)!r}; "
            f"received {peer!r}"
        )
    if getattr(executor, "config", None) != config:
        raise ValueError("compiled P2P executor compression config does not match")
    if group is not None and getattr(executor, "group", None) is not group:
        raise ValueError("compiled P2P executor process group does not match")
    if tag != 0 and getattr(executor, "tag", None) != tag:
        raise ValueError("compiled P2P executor tag does not match")
