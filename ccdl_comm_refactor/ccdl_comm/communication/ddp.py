from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.error_feedback import ErrorFeedbackState


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
    error_feedback: ErrorFeedbackState = field(default_factory=ErrorFeedbackState)

    def process(self, bucket: Any, *, dtype: str) -> Any:
        key = _bucket_key(bucket)
        original = _bucket_tensor(bucket)
        prepared = self.error_feedback.compensate(key, original) if self.config.error_feedback else original

        payload = self.quantize(prepared, self.config)
        restored = self.dequantize(payload, _tensor_shape(prepared), self.config, dtype)

        if self.config.error_feedback:
            self.error_feedback.update(key, original=prepared, transmitted=restored)
        return restored
