"""Optional Ascend CANN backend support."""

from .codec import dequantize_tensor_cann, quantize_tensor_cann
from .diagnostics import CannCapabilityReport, detect_cann
from .loader import CannExtensionStatus, load_cann_extension

__all__ = [
    "CannCapabilityReport",
    "CannExtensionStatus",
    "dequantize_tensor_cann",
    "detect_cann",
    "load_cann_extension",
    "quantize_tensor_cann",
]
