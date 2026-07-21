"""CCDL communication compression library.

This package is the refactored, ParaScale-facing communication layer.  It keeps
training orchestration outside CCDL and exposes a small plugin-oriented API for
native DDP gradient compression.
"""

from .capability import CapabilityReport, detect
from .config import CompressionConfig
from .plugin import CCDLCommunicationPlugin

__all__ = [
    "CapabilityReport",
    "CCDLCommunicationPlugin",
    "CompressionConfig",
    "detect",
]
