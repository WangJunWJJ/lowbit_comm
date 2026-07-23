"""Optional Ascend CANN backend support."""

from .codec import dequantize_tensor_cann, quantize_tensor_cann
from .loader import CannExtensionStatus, load_cann_extension

__all__ = ["CannExtensionStatus", "dequantize_tensor_cann", "load_cann_extension", "quantize_tensor_cann"]
