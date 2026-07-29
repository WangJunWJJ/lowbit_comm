"""CCDL communication compression library.

This package is the refactored, ParaScale-facing communication layer.  It keeps
training orchestration outside CCDL and exposes a small plugin-oriented API for
native DDP gradient compression.
"""

from .capability import CapabilityReport, detect
from .collectives import (
    CollectiveWork,
    CompletionWork,
    ImmediateWork,
    ReducedShard,
    compressed_all_gather,
    compressed_all_gather_dynamic,
    compressed_all_reduce,
    compressed_hierarchical_all_reduce,
    compressed_reduce_scatter,
    compressed_reduce_scatter_shard,
    qall_gather_dyn,
)
from .config import CompressionConfig
from .exceptions import CCDLError, CCDLUnavailableError, TorchDistributedUnavailableError, UnsupportedCollective
from .communication import iqrecv, iqsend, qrecv, qsend
from .plugin import CCDLCommunicationPlugin
from .quantization import Quantizer, dequantize_tensor, estimate_quantized_size, quantize_tensor

__all__ = [
    "CCDLError",
    "CCDLUnavailableError",
    "CapabilityReport",
    "CCDLCommunicationPlugin",
    "CompressionConfig",
    "CollectiveWork",
    "CompletionWork",
    "ImmediateWork",
    "ReducedShard",
    "Quantizer",
    "TorchDistributedUnavailableError",
    "UnsupportedCollective",
    "compressed_all_gather",
    "compressed_all_gather_dynamic",
    "compressed_all_reduce",
    "compressed_hierarchical_all_reduce",
    "compressed_reduce_scatter",
    "compressed_reduce_scatter_shard",
    "dequantize_tensor",
    "detect",
    "estimate_quantized_size",
    "iqrecv",
    "iqsend",
    "quantize_tensor",
    "qall_gather_dyn",
    "qrecv",
    "qsend",
]
