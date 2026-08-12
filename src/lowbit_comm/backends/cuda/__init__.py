"""Optional CUDA backend operations with CPU-safe imports."""

from .backend import CudaBackend
from .codec import ExtensionUnavailable, dequantize_into, payload_nbytes, quantize_into
from .loader import CudaExtensionStatus, load_cuda_extension
from .p2p import CudaQuantizedReceiver, CudaQuantizedSender, P2PTags

__all__ = [
    "CudaExtensionStatus",
    "CudaBackend",
    "CudaQuantizedReceiver",
    "CudaQuantizedSender",
    "ExtensionUnavailable",
    "dequantize_into",
    "load_cuda_extension",
    "payload_nbytes",
    "P2PTags",
    "quantize_into",
]
