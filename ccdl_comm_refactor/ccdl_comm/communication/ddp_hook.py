from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.ddp import DDPBucketProcessor
from ccdl_comm.communication.gather_reduce import CompressedAllGatherReduce, GatheredPayloads
from ccdl_comm.communication.payload_packing import (
    DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    make_fused_payload_all_gather,
    make_payload_all_gather,
    should_fuse_payload,
)
from ccdl_comm.communication.torch_transport import make_torch_all_gather, make_torch_all_reduce
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
    dtype: str = "auto",
    strategy: str = "all_reduce",
    reduce: str = "mean",
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None,
    all_gather: Callable[[Any], GatheredPayloads] | None = None,
    fuse_payload: bool = False,
    fuse_payload_min_numel: int = DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    error_feedback: ErrorFeedbackState | None = None,
    extension_status: CudaExtensionStatus | None = None,
    future_factory: Callable[[], Any] = _torch_future_factory,
    annotation_provider: Callable[[], dict[str, Any]] | None = None,
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

    feedback = error_feedback or ErrorFeedbackState()

    if strategy == "all_gather":
        if all_gather is not None:
            normal_all_gather = all_gather
            fused_all_gather = all_gather
        else:
            buffer_all_gather = make_torch_all_gather()
            normal_all_gather = make_payload_all_gather(buffer_all_gather)
            fused_all_gather = make_fused_payload_all_gather(buffer_all_gather)

        def process_bucket(bucket: Any) -> Any:
            key = bucket.index() if callable(getattr(bucket, "index", None)) else id(bucket)
            original = bucket.buffer()
            prepared = feedback.compensate(key, original) if config.error_feedback else original
            active_dtype = _resolve_dtype(dtype, prepared)
            active_all_gather = (
                fused_all_gather
                if should_fuse_payload(prepared, enabled=fuse_payload, min_numel=fuse_payload_min_numel)
                else normal_all_gather
            )
            collective = CompressedAllGatherReduce(
                config=config,
                compress=active_quantize,
                all_gather=active_all_gather,
                decompress=active_dequantize,
            )
            restored = collective.run(prepared, shape=tuple(prepared.shape), dtype=active_dtype, reduce=reduce)
            if config.error_feedback:
                feedback.update(key, original=prepared, transmitted=restored)
            return restored

    elif strategy == "all_reduce":
        processor = DDPBucketProcessor(
            config=config,
            quantize=active_quantize,
            dequantize=active_dequantize,
            all_reduce=all_reduce or make_torch_all_reduce(),
            error_feedback=feedback,
        )

        def process_bucket(bucket: Any) -> Any:
            tensor = bucket.buffer()
            return processor.process(bucket, dtype=_resolve_dtype(dtype, tensor))

    else:
        raise ValueError(f"unsupported DDP comm hook strategy: {strategy}")

    def hook(state: Any, bucket: Any) -> Any:
        result = process_bucket(bucket)
        future = future_factory()
        future.set_result(result)
        return future

    _apply_ddp_annotations(hook, annotation_provider)
    return hook


def _resolve_dtype(dtype: str, tensor: Any) -> str:
    if dtype != "auto":
        return dtype
    tensor_dtype = str(getattr(tensor, "dtype", ""))
    if "bfloat16" in tensor_dtype:
        return "bf16"
    if "float16" in tensor_dtype or tensor_dtype.endswith("half"):
        return "fp16"
    if "float32" in tensor_dtype or tensor_dtype.endswith("float"):
        return "fp32"
    raise ValueError(f"cannot infer CCDL dtype from bucket tensor dtype: {tensor_dtype!r}")


def _apply_ddp_annotations(hook: Callable[[Any, Any], Any], provider: Callable[[], dict[str, Any]] | None) -> None:
    if provider is None:
        provider = _torch_ddp_annotations
    try:
        hook.__annotations__ = provider()
    except (ImportError, ModuleNotFoundError, AttributeError):
        return


def _torch_ddp_annotations() -> dict[str, Any]:
    torch = import_module("torch")
    dist = import_module("torch.distributed")
    return {
        "state": object,
        "bucket": dist.GradBucket,
        "return": torch.futures.Future[torch.Tensor],
    }
