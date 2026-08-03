"""CCDL communication compression library.

This package is the refactored, ParaScale-facing communication layer.  It keeps
training orchestration outside CCDL and exposes a small plugin-oriented API for
native DDP gradient compression.
"""

from .backend import (
    AutoStrategySelector,
    BackendCapabilities,
    CommunicationBackend,
    StrategyChoice,
)
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
from .compiler import CompileCache, ResolvedPlan, compile, resolve_plan
from .execution_info import ExecutionCounterSnapshot, ExecutionCounters, ExecutionInfo
from .exceptions import (
    BackendRegistrationError,
    CCDLError,
    CCDLUnavailableError,
    TorchDistributedUnavailableError,
    UnsupportedCollective,
)
from .executor import CompileCacheKey, CompiledCommunicationPlan, CompiledExecutor
from .plan import CommunicationPlan, CompileContext, WorkspacePolicy
from .stage import CommunicationStage
from .registry import BackendKey, BackendRegistry
from .communication import iqrecv, iqsend, qrecv, qsend
from .plugin import CCDLCommunicationPlugin
from .quantization import Quantizer, dequantize_tensor, estimate_quantized_size, quantize_tensor

__all__ = [
    "CCDLError",
    "BackendCapabilities",
    "AutoStrategySelector",
    "BackendKey",
    "BackendRegistrationError",
    "BackendRegistry",
    "CCDLUnavailableError",
    "CapabilityReport",
    "CCDLCommunicationPlugin",
    "CompressionConfig",
    "CompileCache",
    "CompileCacheKey",
    "CompiledCommunicationPlan",
    "CommunicationPlan",
    "CommunicationStage",
    "CommunicationBackend",
    "CompileContext",
    "CollectiveWork",
    "CompletionWork",
    "CompiledExecutor",
    "ImmediateWork",
    "ExecutionInfo",
    "ExecutionCounterSnapshot",
    "ExecutionCounters",
    "ReducedShard",
    "Quantizer",
    "TorchDistributedUnavailableError",
    "StrategyChoice",
    "UnsupportedCollective",
    "WorkspacePolicy",
    "compile",
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
    "ResolvedPlan",
    "resolve_plan",
]
