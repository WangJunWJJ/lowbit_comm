"""CUDA executors that own pre-bound production communication operations."""

from __future__ import annotations

from collections.abc import Callable

from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.work import CollectiveWork, bind_execution_work


class CudaAllReduceExecutor:
    """Execute a pre-bound CUDA all-reduce operation."""

    def __init__(self, operation: Callable[[object], object], execution_info: ExecutionInfo) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.workspace_pool = getattr(operation, "workspace_pool", None)
        self.execution_info = execution_info
        self.execution_counters = ExecutionCounters()

    def run(self, tensor: object) -> CollectiveWork[object]:
        self.execution_counters._record_run()
        try:
            result = self._operation(tensor)
            return bind_execution_work(result, self.execution_info, self.execution_counters)
        except BaseException:
            self.execution_counters._record_failed()
            raise


class CudaReducedShardExecutor:
    """Execute a pre-bound CUDA collective returning this rank's shard."""

    def __init__(self, operation: Callable[[object], object], execution_info: ExecutionInfo) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.workspace_pool = getattr(operation, "workspace_pool", None)
        self.execution_info = execution_info
        self.execution_counters = ExecutionCounters()

    def run(self, tensor: object) -> CollectiveWork[object]:
        self.execution_counters._record_run()
        try:
            result = self._operation(tensor)
            return bind_execution_work(result, self.execution_info, self.execution_counters)
        except BaseException:
            self.execution_counters._record_failed()
            raise
