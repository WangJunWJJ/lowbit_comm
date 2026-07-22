"""CCDL communication compression library.

This package is the refactored, ParaScale-facing communication layer.  It keeps
training orchestration outside CCDL and exposes a small plugin-oriented API for
native DDP gradient compression.
"""

from .capability import CapabilityReport, detect
from .config import CompressionConfig
from .exceptions import CCDLError, CCDLUnavailableError, TorchDistributedUnavailableError, UnsupportedCollective
from .plugin import CCDLCommunicationPlugin
from .quantization import dequantize_tensor, estimate_quantized_size, quantize_tensor

__all__ = [
    "CCDLError",
    "CCDLUnavailableError",
    "CapabilityReport",
    "CCDLCommunicationPlugin",
    "CompressionConfig",
    "TorchDistributedUnavailableError",
    "UnsupportedCollective",
    "dequantize_tensor",
    "detect",
    "estimate_quantized_size",
    "quantize_tensor",
]
