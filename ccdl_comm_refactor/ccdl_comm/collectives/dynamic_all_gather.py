from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor


def compile_dynamic_all_gather(**kwargs: Any) -> Any:
    """Compile a bounded reusable dynamic all-gather executor."""

    from ccdl_comm.cuda.dynamic_gather_executor import (
        compile_dynamic_all_gather as compile_cuda_dynamic_all_gather,
    )

    return compile_cuda_dynamic_all_gather(**kwargs)


def compressed_all_gather_dynamic(
    tensor: Any,
    *,
    config: CompressionConfig,
    dtype: str = "auto",
    extension_status: Any | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize: Callable[..., Any] = dequantize_tensor,
    compiled_executor: Any | None = None,
) -> list[Any]:
    """Gather compressed tensors with rank-local dynamic shapes."""

    if compiled_executor is not None:
        if getattr(compiled_executor, "config", None) != config:
            raise ValueError(
                "compiled dynamic all-gather compression config does not match"
            )
        if dtype != "auto" and getattr(compiled_executor, "dtype", None) != dtype:
            raise ValueError("compiled dynamic all-gather dtype does not match")
        return compiled_executor.run(tensor).wait()

    dist = _distributed(import_module_fn)
    torch = import_module_fn("torch")
    active_dtype = _resolve_dtype(dtype, tensor)
    payload = quantize(tensor, config, extension_status=extension_status)
    local_meta = {
        "shape": tuple(tensor.shape),
        "dtype": active_dtype,
        "payload_numel": int(payload.numel()),
    }
    world_size = int(dist.get_world_size())
    metadata: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(metadata, local_meta)
    max_payload_numel = max(int(item["payload_numel"]) for item in metadata if item is not None)
    padded_payload = _pad_payload(payload, max_payload_numel, torch)
    gathered = [padded_payload.new_empty((max_payload_numel,)) for _ in range(world_size)]
    dist.all_gather(gathered, padded_payload)
    result = []
    for buffer, meta in zip(gathered, metadata, strict=True):
        if meta is None:
            raise RuntimeError("missing dynamic all-gather metadata")
        payload_numel = int(meta["payload_numel"])
        trimmed = buffer[:payload_numel]
        result.append(
            dequantize(
                trimmed,
                tuple(meta["shape"]),
                config,
                dtype=str(meta["dtype"]),
                extension_status=extension_status,
                reduce_op="none",
            )
        )
    return result


def qall_gather_dyn(tensor: Any, *, config: CompressionConfig, **kwargs: Any) -> list[Any]:
    """Compatibility-style alias for dynamic compressed all-gather."""

    return compressed_all_gather_dynamic(tensor, config=config, **kwargs)


def _pad_payload(payload: Any, target_numel: int, torch: Any) -> Any:
    payload_numel = int(payload.numel())
    if payload_numel == target_numel:
        return payload
    if payload_numel > target_numel:
        raise ValueError("payload larger than dynamic all-gather target size")
    padding = target_numel - payload_numel
    zeros = payload.new_zeros((padding,))
    return torch.cat((payload, zeros), dim=0)


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
