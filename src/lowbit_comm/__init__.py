"""Typed low-bit communication library."""

from .compiler import BackendRegistry, BenchmarkEvidence, compile
from .core import (
    AutoAlgorithm,
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    ErrorFeedbackDomain,
    FullPrecisionWire,
    FullTensor,
    HierarchicalCompressed,
    NativeAllReduce,
    QuantizedWire,
    ReduceMean,
    ReduceSum,
    ReducedShard,
    RuntimeBindings,
)

__version__ = "0.3.0"

__all__ = [
    "AutoAlgorithm",
    "BackendRegistry",
    "BenchmarkEvidence",
    "CommunicationProgram",
    "CompileContext",
    "CompressedAllGather",
    "CompressedReduceScatter",
    "CompressedReduceScatterAllGather",
    "DataType",
    "ErrorFeedbackDomain",
    "FullPrecisionWire",
    "FullTensor",
    "HierarchicalCompressed",
    "NativeAllReduce",
    "QuantizedWire",
    "ReduceMean",
    "ReduceSum",
    "ReducedShard",
    "RuntimeBindings",
    "__version__",
    "compile",
]
