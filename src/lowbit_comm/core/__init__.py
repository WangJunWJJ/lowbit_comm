"""Stable Semantic IR public types."""

from .backend import (
    BackendCapabilities,
    CapabilitySpec,
    CommunicationBackend,
    CompiledExecutable,
)
from .context import CompileContext, RuntimeBindings
from .errors import LowBitCommError, ProgramVerificationError, UnsupportedProgram
from .execution_info import ExecutionInfo
from .lowered import (
    BufferPlan,
    BufferSpec,
    ExecutorKind,
    LoweredProgram,
    LoweredStage,
    StageKind,
    WorkspaceRole,
)
from .metadata import MetadataPacket
from .operations import ReductionContract, ReduceMean, ReduceSum, compile_reduction
from .primitives import PhysicalPrimitive
from .program import CommunicationProgram
from .topology import GroupedReductionPlan, Topology
from .types import (
    AutoAlgorithm,
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
    ReducedShard,
)
from .values import ReducedShardValue

__all__ = [
    "AutoAlgorithm",
    "BackendCapabilities",
    "BufferPlan",
    "BufferSpec",
    "CapabilitySpec",
    "CompileContext",
    "CommunicationProgram",
    "CompressedAllGather",
    "CompressedReduceScatter",
    "CompressedReduceScatterAllGather",
    "CommunicationBackend",
    "CompiledExecutable",
    "DataType",
    "ErrorFeedbackDomain",
    "ExecutionInfo",
    "ExecutorKind",
    "FullPrecisionWire",
    "FullTensor",
    "GroupedReductionPlan",
    "HierarchicalCompressed",
    "LowBitCommError",
    "LoweredProgram",
    "LoweredStage",
    "MetadataPacket",
    "NativeAllReduce",
    "PhysicalPrimitive",
    "ProgramVerificationError",
    "QuantizedWire",
    "ReductionContract",
    "ReduceMean",
    "ReduceSum",
    "ReducedShard",
    "ReducedShardValue",
    "RuntimeBindings",
    "StageKind",
    "Topology",
    "UnsupportedProgram",
    "WorkspaceRole",
    "compile_reduction",
]
