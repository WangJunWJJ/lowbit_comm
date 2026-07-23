"""Optional Ascend CANN backend support."""

from .loader import CannExtensionStatus, load_cann_extension

__all__ = ["CannExtensionStatus", "load_cann_extension"]
