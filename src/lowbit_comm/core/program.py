"""Immutable Semantic IR root."""

from __future__ import annotations

from dataclasses import dataclass

from .operations import ReduceMean, ReduceSum
from .types import (
    AutoAlgorithm,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    ErrorFeedbackDomain,
    FullPrecisionWire,
    FullTensor,
    HierarchicalCompressed,
    NativeAllReduce,
    QuantizedWire,
    ReducedShard,
)


Operation = ReduceMean | ReduceSum
Output = FullTensor | ReducedShard
Wire = FullPrecisionWire | QuantizedWire
Algorithm = (
    AutoAlgorithm
    | NativeAllReduce
    | CompressedAllGather
    | CompressedReduceScatter
    | CompressedReduceScatterAllGather
    | HierarchicalCompressed
)


@dataclass(frozen=True, slots=True)
class CommunicationProgram:
    operation: Operation
    output: Output
    wire: Wire
    algorithm: Algorithm
    async_op: bool = True
    error_feedback: ErrorFeedbackDomain = ErrorFeedbackDomain.NONE

    def __post_init__(self) -> None:
        for name in ("operation", "output", "wire", "algorithm"):
            if getattr(self, name) is None:
                raise TypeError(f"{name} must not be None")
        if not isinstance(self.async_op, bool):
            raise TypeError("async_op must be a bool")
        if not isinstance(self.error_feedback, ErrorFeedbackDomain):
            raise TypeError("error_feedback must be an ErrorFeedbackDomain")
