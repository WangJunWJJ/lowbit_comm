"""CUDA executors that own pre-bound production communication operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ccdl_comm.execution_info import ExecutionInfo
from ccdl_comm.work import CollectiveWork, ImmediateWork


class CudaAllReduceExecutor:
    """Execute a pre-bound CUDA all-reduce operation."""

    def __init__(self, operation: Callable[[object], object], execution_info: ExecutionInfo) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.execution_info = execution_info

    def run(self, tensor: object) -> CollectiveWork[object]:
        result = self._operation(tensor)
        return _as_work(result)


class CudaReducedShardExecutor:
    """Execute a pre-bound CUDA collective returning this rank's shard."""

    def __init__(self, operation: Callable[[object], object], execution_info: ExecutionInfo) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.execution_info = execution_info

    def run(self, tensor: object) -> CollectiveWork[object]:
        result = self._operation(tensor)
        return _as_work(result)


def _as_work(result: Any) -> CollectiveWork[Any]:
    if isinstance(result, CollectiveWork):
        return result
    if callable(getattr(result, "wait", None)) and callable(getattr(result, "query", None)):
        return result
    return ImmediateWork(result)
