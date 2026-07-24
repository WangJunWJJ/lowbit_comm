from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus, load_cuda_extension
from ccdl_comm.exceptions import CCDLUnavailableError


@dataclass(frozen=True)
class _ExtensionSymbols:
    module: object
    quant_type: object
    dtype: object | None = None
    reduce_op: object | None = None


_QUANT_TYPE_ATTRS = {
    "linear": "Linear",
    "normal": "Normal",
    "uniform": "Uniform",
    "e3m0": "E3M0",
    "e2m1": "E2M1",
}

_DTYPE_ATTRS = {
    "fp16": "FP16",
    "bf16": "BF16",
    "fp32": "FP32",
}


def _extension_status_or_default(extension_status: CudaExtensionStatus | None) -> CudaExtensionStatus:
    if extension_status is not None:
        return extension_status
    return load_cuda_extension()


def _require_available_extension(extension_status: CudaExtensionStatus | None) -> object:
    status = _extension_status_or_default(extension_status)
    if not status.available or status.module is None:
        raise CCDLUnavailableError(status.reason or "ccdl_cuda_ops is not available")
    return status.module


def _get_required_attr(obj: object, attr_name: str) -> object:
    if not hasattr(obj, attr_name):
        raise CCDLUnavailableError(f"ccdl_cuda_ops missing required symbol: {attr_name}")
    return getattr(obj, attr_name)


def _get_quant_type(module: object, quant_type: str) -> object:
    enum = _get_required_attr(module, "QuantType")
    attr_name = _QUANT_TYPE_ATTRS[quant_type]
    return _get_required_attr(enum, attr_name)


def _get_dtype(module: object, dtype: str) -> object:
    enum = _get_required_attr(module, "DType")
    attr_name = _DTYPE_ATTRS[dtype]
    return _get_required_attr(enum, attr_name)


def _get_reduce_op(module: object, reduce_op: str) -> object:
    enum = _get_required_attr(module, "ReduceOP")
    attr_name = reduce_op.upper()
    return _get_required_attr(enum, attr_name)


def quantize_tensor(
    tensor: object,
    config: CompressionConfig,
    *,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Quantize a tensor through the CCDL CUDA extension.

    Args:
        tensor: Tensor-like object accepted by `ccdl_cuda_ops.quantize`.
        config: Compression policy.
        extension_status: Optional preloaded extension status for tests or
            planner-controlled execution.

    Returns:
        The quantized tensor buffer returned by the CUDA extension.

    Raises:
        CCDLUnavailableError: If the CUDA extension is missing or incomplete.
    """

    module = _require_available_extension(extension_status)
    quantize = _get_required_attr(module, "quantize")
    quant_type = _get_quant_type(module, config.quant_type)
    padded_tensor = _pad_tensor_to_group_size(tensor, config.group_size)
    return quantize(
        padded_tensor,
        config.group_size,
        config.topk,
        config.stochastic,
        config.bit,
        quant_type,
        config.compact,
    )


def dequantize_tensor(
    buffer: object,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Dequantize a tensor buffer through the CCDL CUDA extension.

    Args:
        buffer: Quantized tensor buffer returned by CCDL.
        shape: Original tensor shape.
        config: Compression policy.
        dtype: Original tensor dtype name: `fp16`, `bf16`, or `fp32`.
        extension_status: Optional preloaded extension status for tests or
            planner-controlled execution.

    Returns:
        The dequantized tensor reshaped to `shape` when the returned object
        provides a `reshape` method.

    Raises:
        CCDLUnavailableError: If the CUDA extension is missing or incomplete.
    """

    module = _require_available_extension(extension_status)
    dequantize = _get_required_attr(module, "dequantize")
    quant_type = _get_quant_type(module, config.quant_type)
    dtype_enum = _get_dtype(module, dtype)
    reduce_op = _get_reduce_op(module, "none")
    decoded = dequantize(
        buffer,
        config.group_size,
        config.topk,
        config.bit,
        reduce_op,
        quant_type,
        dtype_enum,
        config.compact,
    )
    if hasattr(decoded, "reshape"):
        original_numel = _numel(shape)
        flattened = decoded.reshape((-1,))
        try:
            trimmed = flattened[:original_numel]
        except TypeError:
            trimmed = flattened
        return trimmed.reshape(shape)
    return decoded


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _pad_tensor_to_group_size(tensor: Any, group_size: int, *, torch_module: Any | None = None) -> Any:
    if not hasattr(tensor, "numel") or not hasattr(tensor, "reshape"):
        return tensor
    numel = tensor.numel()
    remainder = numel % group_size
    if remainder == 0:
        return tensor
    padding = group_size - remainder
    flat = tensor.reshape((-1,))
    zeros = flat.new_zeros((padding,))
    torch = torch_module or import_module("torch")
    return torch.cat((flat, zeros), dim=0)
