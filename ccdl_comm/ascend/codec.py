from __future__ import annotations

from typing import Any

from ccdl_comm.ascend.loader import CannExtensionStatus, load_cann_extension
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import CCDLUnavailableError


def quantize_tensor_cann(
    tensor: Any,
    config: CompressionConfig,
    *,
    extension_status: CannExtensionStatus | None = None,
) -> CompressedPayload:
    """Quantize a tensor through the optional CANN extension."""

    _validate_supported_config(config)
    module = _require_available_extension(extension_status)
    quantize = _get_required_attr(module, "quantize_linear_int8")
    result = quantize(tensor, config.group_size)
    return CompressedPayload(
        buffer=_get_required_attr(result, "buffer"),
        shape=tuple(getattr(tensor, "shape", ())),
        dtype=_resolve_dtype(tensor),
        metadata={
            "scales": _get_required_attr(result, "scales"),
            "original_numel": int(_get_required_attr(result, "original_numel")),
        },
    )


def dequantize_tensor_cann(
    payload: CompressedPayload,
    shape: tuple[int, ...],
    config: CompressionConfig,
    dtype: str,
    *,
    extension_status: CannExtensionStatus | None = None,
) -> Any:
    """Dequantize a tensor through the optional CANN extension."""

    _validate_supported_config(config)
    module = _require_available_extension(extension_status)
    dequantize = _get_required_attr(module, "dequantize_linear_int8")
    return dequantize(
        payload.buffer,
        payload.metadata["scales"],
        int(payload.metadata["original_numel"]),
        shape,
        dtype,
        config.group_size,
    )


def _require_available_extension(extension_status: CannExtensionStatus | None) -> object:
    status = extension_status or load_cann_extension()
    if not status.available or status.module is None:
        raise CCDLUnavailableError(status.reason or "ccdl_cann_ops is not available")
    return status.module


def _get_required_attr(obj: object, attr_name: str) -> Any:
    if not hasattr(obj, attr_name):
        raise CCDLUnavailableError(f"ccdl_cann_ops missing required symbol: {attr_name}")
    return getattr(obj, attr_name)


def _validate_supported_config(config: CompressionConfig) -> None:
    if config.bit != 8:
        raise ValueError("CANN codec only supports bit=8")
    if config.quant_type != "linear":
        raise ValueError("CANN codec only supports quant_type='linear'")
    if config.topk != 0:
        raise ValueError("CANN codec only supports topk=0")


def _resolve_dtype(tensor: Any) -> str:
    tensor_dtype = str(getattr(tensor, "dtype", ""))
    if "bfloat16" in tensor_dtype:
        return "bf16"
    if "float16" in tensor_dtype or tensor_dtype.endswith("half"):
        return "fp16"
    if "float32" in tensor_dtype or tensor_dtype.endswith("float"):
        return "fp32"
    raise ValueError(f"cannot infer CCDL dtype from tensor dtype: {tensor_dtype!r}")
