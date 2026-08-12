from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor
from ccdl_comm.quantization.error_feedback import ErrorFeedbackState
from ccdl_comm.reduction import ReductionContract


def _bucket_key(bucket: Any) -> Hashable:
    index = getattr(bucket, "index", None)
    if callable(index):
        return index()
    return id(bucket)


def _bucket_tensor(bucket: Any) -> Any:
    buffer = getattr(bucket, "buffer", None)
    if callable(buffer):
        return buffer()
    raise TypeError("DDP bucket object must provide a callable buffer() method")


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", ())
    return tuple(shape)


@dataclass
class DDPBucketProcessor:
    """Synchronous compressed bucket processor used to build DDP hooks.

    This class intentionally does not call ``torch.distributed``.  It only
    defines the local bucket preparation and reconstruction flow so ParaScale can
    wrap it inside native-DDP communication hooks.
    """

    config: CompressionConfig
    quantize: Callable[[Any, CompressionConfig], Any]
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any]
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None
    error_feedback: ErrorFeedbackState = field(default_factory=ErrorFeedbackState)

    @classmethod
    def from_cuda_codec(
        cls,
        config: CompressionConfig,
        *,
        extension_status: CudaExtensionStatus | None = None,
        error_feedback: ErrorFeedbackState | None = None,
    ) -> DDPBucketProcessor:
        def quantize(tensor: Any, active_config: CompressionConfig) -> Any:
            return quantize_tensor(tensor, active_config, extension_status=extension_status)

        def dequantize(payload: Any, shape: tuple[int, ...], active_config: CompressionConfig, dtype: str) -> Any:
            return dequantize_tensor(payload, shape, active_config, dtype=dtype, extension_status=extension_status)

        return cls(
            config=config,
            quantize=quantize,
            dequantize=dequantize,
            error_feedback=error_feedback or ErrorFeedbackState(),
        )

    def process(
        self,
        bucket: Any,
        *,
        dtype: str,
        reduction: ReductionContract | None = None,
    ) -> Any:
        key = _bucket_key(bucket)
        original = _bucket_tensor(bucket)
        prepared = self.error_feedback.compensate(key, original) if self.config.error_feedback else original

        if self.all_reduce is None:
            payload = self.quantize(prepared, self.config)
            restored = self.dequantize(payload, _tensor_shape(prepared), self.config, dtype)
            local_restored = restored if self.config.error_feedback else None
        else:
            active_reduction = reduction or ReductionContract(op="sum", world_size=1)
            local_payload = CompressedPayload(
                buffer=self.quantize(prepared, self.config),
                shape=_tensor_shape(prepared),
                dtype=dtype,
            )
            reduced_payload = self.all_reduce(local_payload, active_reduction.transport_op)
            restored = self.dequantize(
                reduced_payload.buffer,
                reduced_payload.shape,
                self.config,
                reduced_payload.dtype,
            )
            restored = active_reduction.normalize(restored)
            local_restored = None
            if self.config.error_feedback:
                local_restored = self.dequantize(
                    local_payload.buffer,
                    local_payload.shape,
                    self.config,
                    local_payload.dtype,
                )

        if self.config.error_feedback:
            if local_restored is None:
                raise RuntimeError("local reconstruction is required for error feedback")
            self.error_feedback.update_local(key, prepared=prepared, local_restored=local_restored)
        return restored
