from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.ddp import DDPBucketProcessor
from ccdl_comm.communication.torch_transport import make_torch_all_reduce
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor
from ccdl_comm.quantization.error_feedback import ErrorFeedbackState


def _torch_future_factory() -> Any:
    torch = import_module("torch")
    return torch.futures.Future()


def create_ddp_comm_hook(
    config: CompressionConfig,
    *,
    dtype: str,
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None,
    error_feedback: ErrorFeedbackState | None = None,
    extension_status: CudaExtensionStatus | None = None,
    future_factory: Callable[[], Any] = _torch_future_factory,
) -> Callable[[Any, Any], Any]:
    """Create a PyTorch DDP comm hook backed by CCDL bucket processing."""

    def active_quantize(tensor: Any, active_config: CompressionConfig) -> Any:
        if quantize is not None:
            return quantize(tensor, active_config)
        return quantize_tensor(tensor, active_config, extension_status=extension_status)

    def active_dequantize(payload: Any, shape: tuple[int, ...], active_config: CompressionConfig, active_dtype: str) -> Any:
        if dequantize is not None:
            return dequantize(payload, shape, active_config, active_dtype)
        return dequantize_tensor(payload, shape, active_config, dtype=active_dtype, extension_status=extension_status)

    processor = DDPBucketProcessor(
        config=config,
        quantize=active_quantize,
        dequantize=active_dequantize,
        all_reduce=all_reduce or make_torch_all_reduce(),
        error_feedback=error_feedback or ErrorFeedbackState(),
    )

    def hook(state: Any, bucket: Any) -> Any:
        result = processor.process(bucket, dtype=dtype)
        future = future_factory()
        future.set_result(result)
        return future

    return hook
