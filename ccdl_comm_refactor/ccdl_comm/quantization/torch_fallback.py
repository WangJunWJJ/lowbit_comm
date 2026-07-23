from __future__ import annotations

from importlib import import_module
from typing import Any

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig


def quantize_tensor_fallback(tensor: Any, config: CompressionConfig) -> CompressedPayload:
    """Quantize a tensor with portable torch ops.

    This fallback is intended for correctness and non-CUDA device enablement
    such as Ascend NPU validation. High-performance CUDA keeps using the native
    extension path.
    """

    if config.quant_type != "linear":
        raise ValueError("torch fallback supports only quant_type='linear'")
    if config.bit != 8:
        raise ValueError("torch fallback only supports bit=8")

    torch = import_module("torch")
    flat = tensor.reshape((-1,))
    original_numel = flat.numel()
    padded = _pad_flat_tensor(flat, config.group_size, torch)
    groups = padded.reshape((-1, config.group_size)).float()
    max_abs = groups.abs().amax(dim=1)
    scales = torch.clamp(max_abs / 127.0, min=torch.finfo(torch.float32).eps)
    quantized = torch.round(groups / scales.reshape((-1, 1))).clamp(-127, 127).to(torch.int8)
    return CompressedPayload(
        buffer=quantized.reshape((-1,)),
        shape=tuple(getattr(tensor, "shape", ())),
        dtype=_ccdl_dtype(tensor),
        metadata={"scales": scales, "original_numel": original_numel},
    )


def dequantize_tensor_fallback(
    payload: CompressedPayload,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
) -> Any:
    """Dequantize a tensor produced by :func:`quantize_tensor_fallback`."""

    torch_dtype = _torch_dtype(dtype)
    groups = payload.buffer.reshape((-1, config.group_size)).float()
    scales = payload.metadata["scales"]
    original_numel = int(payload.metadata["original_numel"])
    restored = groups * scales.reshape((-1, 1))
    flat = restored.reshape((-1,))[:original_numel]
    return flat.to(dtype=torch_dtype).reshape(shape)


def _pad_flat_tensor(flat: Any, group_size: int, torch: Any) -> Any:
    remainder = flat.numel() % group_size
    if remainder == 0:
        return flat
    padding = group_size - remainder
    return torch.cat((flat, flat.new_zeros((padding,))), dim=0)


def _torch_dtype(dtype: str) -> Any:
    torch = import_module("torch")
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported torch fallback dtype: {dtype!r}")


def _ccdl_dtype(tensor: Any) -> str:
    tensor_dtype = str(getattr(tensor, "dtype", ""))
    if "bfloat16" in tensor_dtype:
        return "bf16"
    if "float16" in tensor_dtype or tensor_dtype.endswith("half"):
        return "fp16"
    if "float32" in tensor_dtype or tensor_dtype.endswith("float"):
        return "fp32"
    raise ValueError(f"cannot infer CCDL dtype from tensor dtype: {tensor_dtype!r}")
