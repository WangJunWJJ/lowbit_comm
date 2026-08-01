from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.codec import allocate_quantized_buffer, dequantize_tensor, quantize_tensor


class Quantizer:
    """Object-oriented facade over `CompressionConfig` and CCDL codec functions."""

    def __init__(
        self,
        config: CompressionConfig | None = None,
        *,
        dtype: str = "auto",
        extension_status: Any | None = None,
        quantize_fn: Callable[..., Any] = quantize_tensor,
        dequantize_fn: Callable[..., Any] = dequantize_tensor,
        allocate_fn: Callable[..., Any] = allocate_quantized_buffer,
    ) -> None:
        self.config = config or CompressionConfig()
        self.dtype = dtype
        self.extension_status = extension_status
        self._quantize = quantize_fn
        self._dequantize = dequantize_fn
        self._allocate = allocate_fn

    def quantize(self, tensor: Any, output: Any | None = None) -> Any:
        return self._quantize(tensor, self.config, extension_status=self.extension_status, output=output)

    def dequantize(
        self,
        buffer: Any,
        shape: tuple[int, ...],
        output: Any | None = None,
        *,
        reduce_op: str = "none",
        dtype: str | None = None,
    ) -> Any:
        return self._dequantize(
            buffer,
            shape,
            self.config,
            dtype=dtype or self.dtype,
            extension_status=self.extension_status,
            output=output,
            reduce_op=reduce_op,
        )

    def allocate_q(self, tensor: Any, *, dtype: str | None = None) -> Any:
        return self._allocate(tensor, self.config, dtype=dtype or self.dtype)

    def is_quantized(self) -> bool:
        return self.config.bit < 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Quantizer:
        config_data = dict(data["config"])
        return cls(CompressionConfig(**config_data), dtype=str(data.get("dtype", "auto")))
