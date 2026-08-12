from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from math import ceil
from operator import mul
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus, load_cuda_extension
from ccdl_comm.exceptions import CCDLUnavailableError
from ccdl_comm.quantization.sizing import estimate_quantized_size


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


def allocate_quantized_buffer(
    tensor: Any,
    config: CompressionConfig,
    *,
    dtype: str,
    torch_module: Any | None = None,
) -> Any:
    """Allocate a uint8 output buffer for inplace CUDA quantization."""

    torch = torch_module or import_module("torch")
    estimate = estimate_quantized_size(int(tensor.numel()), dtype=dtype, config=config)
    return tensor.new_empty((estimate.quantized_bytes,), dtype=torch.uint8)


def allocate_dequantized_buffer(
    tensor: Any,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    torch_module: Any | None = None,
) -> Any:
    """Allocate a padded output buffer for inplace CUDA dequantization."""

    del torch_module
    padded_numel = ceil(_numel(shape) / config.group_size) * config.group_size if shape else 0
    return tensor.new_empty((padded_numel,), dtype=getattr(tensor, "dtype"))


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
    output: object | None = None,
    residual: object | None = None,
    metadata: dict[str, Any] | None = None,
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
    quant_type = _get_quant_type(module, config.quant_type)
    if output is not None:
        if _inplace_quantize_pack_with_module(tensor, output, residual, config, metadata, module, quant_type):
            return output
        prepared = tensor if residual is None else tensor + residual
        padded_tensor = _pad_tensor_to_group_size(prepared, config.group_size)
        inplace_quantize = _get_required_attr(module, "inplace_quantize")
        inplace_quantize(
            padded_tensor,
            output,
            config.group_size,
            config.topk,
            config.stochastic,
            config.bit,
            quant_type,
            config.compact,
        )
        return output
    prepared = tensor if residual is None else tensor + residual
    padded_tensor = _pad_tensor_to_group_size(prepared, config.group_size)
    quantize = _get_required_attr(module, "quantize")
    return quantize(
        padded_tensor,
        config.group_size,
        config.topk,
        config.stochastic,
        config.bit,
        quant_type,
        config.compact,
    )


def inplace_quantize_pack(
    tensor: object,
    output: object,
    residual: object | None,
    config: CompressionConfig,
    metadata: dict[str, Any] | None = None,
    *,
    extension_status: CudaExtensionStatus | None = None,
) -> bool:
    """Try the allocation-free fused quantization and compact payload path.

    Unsupported policies return ``False`` so a compiled executor can select a
    fallback before communication. Invalid tensor/workspace contracts remain
    hard errors in the native extension.
    """

    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    return _inplace_quantize_pack_with_module(tensor, output, residual, config, metadata, module, quant_type)


def _inplace_quantize_pack_with_module(
    tensor: object,
    output: object,
    residual: object | None,
    config: CompressionConfig,
    metadata: dict[str, Any] | None,
    module: object,
    quant_type: object,
) -> bool:
    native = getattr(module, "inplace_quantize_pack", None)
    used_fused = False
    if native is not None:
        used_fused = bool(
            native(
                tensor,
                output,
                residual,
                config.group_size,
                config.topk,
                config.stochastic,
                config.bit,
                quant_type,
                config.compact,
            )
        )
    if metadata is not None:
        original_numel = int(tensor.numel())
        padded_numel = ceil(original_numel / config.group_size) * config.group_size if original_numel else 0
        metadata.update(
            original_numel=original_numel,
            padded_numel=padded_numel,
            padding_numel=padded_numel - original_numel,
            fused_quant_pack=used_fused,
        )
    return used_fused


def dequantize_tensor(
    buffer: object,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
    extension_status: CudaExtensionStatus | None = None,
    output: object | None = None,
    reduce_op: str = "none",
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
    quant_type = _get_quant_type(module, config.quant_type)
    dtype_enum = _get_dtype(module, dtype)
    reduce_op_enum = _get_reduce_op(module, reduce_op)
    if output is not None:
        inplace_dequantize = _get_required_attr(module, "inplace_dequantize")
        inplace_dequantize(
            buffer,
            output,
            config.group_size,
            config.topk,
            config.bit,
            reduce_op_enum,
            quant_type,
            config.compact,
        )
        decoded = output
    else:
        dequantize = _get_required_attr(module, "dequantize")
        decoded = dequantize(
            buffer,
            config.group_size,
            config.topk,
            config.bit,
            reduce_op_enum,
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


def dequantize_reduce_tensors(
    buffers: list[object],
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
    extension_status: CudaExtensionStatus | None = None,
    output: object | None = None,
    reduce: str = "sum",
) -> object:
    """Dequantize multiple compressed buffers and reduce them into one tensor."""

    if not buffers:
        raise ValueError("buffers must not be empty")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported dequantize-reduce mode: {reduce}")
    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    dtype_enum = _get_dtype(module, dtype)
    used_fused = False
    if output is None:
        dequantize_reduce = _get_required_attr(module, "dequantize_reduce")
        decoded = dequantize_reduce(
            buffers,
            config.group_size,
            config.topk,
            config.bit,
            quant_type,
            dtype_enum,
            config.compact,
        )
    else:
        if hasattr(module, "inplace_dequantize_reduce_mean"):
            used_fused = inplace_dequantize_reduce_mean(
                buffers,
                output,
                config,
                extension_status=extension_status,
                reduce=reduce,
            )
        if not used_fused:
            inplace_dequantize_reduce = _get_required_attr(module, "inplace_dequantize_reduce")
            inplace_dequantize_reduce(
                buffers,
                output,
                config.group_size,
                config.topk,
                config.bit,
                quant_type,
                config.compact,
            )
        decoded = output
    if reduce == "mean" and not used_fused:
        if output is not None:
            decoded.div_(len(buffers))
        else:
            decoded = decoded / len(buffers)
    if hasattr(decoded, "reshape"):
        original_numel = _numel(shape)
        flattened = decoded.reshape((-1,))
        try:
            trimmed = flattened[:original_numel]
        except TypeError:
            trimmed = flattened
        return trimmed.reshape(shape)
    return decoded


def inplace_dequantize_reduce_mean(
    buffers: list[object],
    output: object,
    config: CompressionConfig,
    *,
    extension_status: CudaExtensionStatus | None = None,
    reduce: str = "sum",
) -> bool:
    """Try the fused CUDA dequantize/reduce path in a caller-owned output."""

    if not buffers:
        raise ValueError("buffers must not be empty")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported dequantize-reduce mode: {reduce}")
    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    inplace_fused = _get_required_attr(module, "inplace_dequantize_reduce_mean")
    divisor = len(buffers) if reduce == "mean" else 1
    return bool(
        inplace_fused(
            buffers,
            output,
            config.group_size,
            config.topk,
            config.bit,
            quant_type,
            config.compact,
            divisor,
        )
    )


def dequantize_reduce_update_error_feedback(
    buffers: list[object],
    prepared: object,
    residual: object,
    shape: tuple[int, ...],
    config: CompressionConfig,
    *,
    dtype: str,
    extension_status: CudaExtensionStatus | None = None,
    reduce: str = "sum",
) -> object:
    """Dequantize/reduce buffers and update an error-feedback residual in one native call."""

    if not buffers:
        raise ValueError("buffers must not be empty")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported dequantize-reduce mode: {reduce}")
    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    dtype_enum = _get_dtype(module, dtype)
    combined = _get_required_attr(module, "dequantize_reduce_update_error_feedback")
    divisor = len(buffers) if reduce == "mean" else 1
    decoded = combined(
        buffers,
        prepared,
        residual,
        config.group_size,
        config.topk,
        config.bit,
        quant_type,
        dtype_enum,
        config.compact,
        divisor,
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


def inplace_dequantize_reduce_mean_update_error_feedback(
    buffers: list[object],
    prepared: object,
    restored: object,
    residual: object,
    config: CompressionConfig,
    *,
    extension_status: CudaExtensionStatus | None = None,
    reduce: str = "sum",
) -> bool:
    """Try the workspace-aware fused CUDA dequant/reduce/mean/error-feedback path."""

    if not buffers:
        raise ValueError("buffers must not be empty")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported dequantize-reduce mode: {reduce}")
    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    inplace_fused = _get_required_attr(module, "inplace_dequantize_reduce_mean_update_error_feedback")
    divisor = len(buffers) if reduce == "mean" else 1
    return bool(
        inplace_fused(
            buffers,
            prepared,
            restored,
            residual,
            config.group_size,
            config.topk,
            config.bit,
            quant_type,
            config.compact,
            divisor,
        )
    )


def inplace_dequantize_reduce_update_local_feedback(
    buffers: list[object],
    local_input_index: int,
    prepared: object,
    restored: object,
    residual: object,
    config: CompressionConfig,
    *,
    extension_status: CudaExtensionStatus | None = None,
    reduce: str = "sum",
) -> bool:
    """Fuse global reduction with this rank's local reconstruction residual."""

    if not buffers:
        raise ValueError("buffers must not be empty")
    if local_input_index < 0 or local_input_index >= len(buffers):
        raise ValueError("local_input_index must identify one gathered payload")
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported dequantize-reduce mode: {reduce}")
    module = _require_available_extension(extension_status)
    quant_type = _get_quant_type(module, config.quant_type)
    inplace_fused = _get_required_attr(
        module,
        "inplace_dequantize_reduce_update_local_error_feedback",
    )
    divisor = len(buffers) if reduce == "mean" else 1
    return bool(
        inplace_fused(
            buffers,
            local_input_index,
            prepared,
            restored,
            residual,
            config.group_size,
            config.topk,
            config.bit,
            quant_type,
            config.compact,
            divisor,
        )
    )


def update_error_feedback_residual(
    prepared: object,
    restored: object,
    residual: object,
    *,
    extension_status: CudaExtensionStatus | None = None,
) -> object:
    """Update an error-feedback residual buffer through the CUDA extension."""

    module = _require_available_extension(extension_status)
    inplace_update = _get_required_attr(module, "inplace_error_feedback_update")
    inplace_update(prepared, restored, residual)
    return residual


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
