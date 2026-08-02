"""CUDA executors that own pre-bound production communication operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.shard import ReducedShard
from ccdl_comm.work import CollectiveWork, bind_execution_work


@dataclass(frozen=True, slots=True)
class PrecollectedPayloadExecution:
    """Result of reducing already-collected compressed payloads."""

    output: object
    fused: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.fused and self.fallback_reason is not None:
            raise ValueError("fused execution must not provide a fallback reason")
        if not self.fused and not self.fallback_reason:
            raise ValueError("fallback execution requires a fallback reason")


class CudaAllReduceExecutor:
    """Execute a pre-bound CUDA all-reduce operation."""

    def __init__(
        self,
        operation: Callable[[object], object],
        execution_info: ExecutionInfo,
        *,
        precollected_operation: Callable[..., PrecollectedPayloadExecution | str | None] | None = None,
    ) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self._precollected_operation = precollected_operation
        self.workspace_pool = getattr(operation, "workspace_pool", None)
        self.execution_info = execution_info
        self.last_execution_info = execution_info
        self._fused_execution_info = replace(
            execution_info,
            fallback_used=False,
            fallback_reason=None,
            fast_path="cuda_fused_dequant_reduce_mean_ef",
        )
        self.execution_counters = ExecutionCounters()

    def run(self, tensor: object) -> CollectiveWork[object]:
        self.execution_counters._record_run()
        try:
            result = self._operation(tensor)
            return bind_execution_work(result, self.execution_info, self.execution_counters)
        except BaseException:
            self.execution_counters._record_failed()
            raise

    def run_precollected_payloads(
        self,
        payloads: object,
        prepared: object,
        output: object,
        residual: object,
    ) -> object:
        """Reduce compressed rank payloads directly into caller-owned workspaces."""

        operation = self._precollected_operation
        if operation is None:
            raise RuntimeError("precollected payload operation was not bound at compile time")
        self.execution_counters._record_run()
        try:
            execution = operation(
                payloads,
                prepared=prepared,
                output=output,
                residual=residual,
            )
            if execution is None:
                self.last_execution_info = self._fused_execution_info
                result = output
            elif isinstance(execution, str):
                if not execution:
                    raise ValueError("fallback reason must not be empty")
                self.last_execution_info = replace(
                    self.execution_info,
                    fallback_used=True,
                    fallback_reason=execution,
                    fast_path="python_fallback",
                )
                result = output
            elif isinstance(execution, PrecollectedPayloadExecution):
                result = execution.output
                if execution.fused:
                    self.last_execution_info = self._fused_execution_info
                else:
                    self.last_execution_info = replace(
                        self.execution_info,
                        fallback_used=True,
                        fallback_reason=execution.fallback_reason,
                        fast_path="python_fallback",
                    )
            else:
                raise TypeError(
                    "precollected operation must return None, a fallback reason, "
                    "or PrecollectedPayloadExecution"
                )
            self.execution_counters._record_completed()
            return result
        except BaseException:
            self.execution_counters._record_failed()
            raise


class CompressedReduceScatterExecutor:
    """Execute a pre-bound compressed reduce-scatter returning a local shard."""

    def __init__(self, operation: Callable[[object], object], execution_info: ExecutionInfo) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.workspace_pool = getattr(operation, "workspace_pool", None)
        self.chunk_plan = getattr(operation, "chunk_plan", None)
        self.execution_info = execution_info
        self.execution_counters = ExecutionCounters()

    def run(self, tensor: object, *, out: object | None = None) -> CollectiveWork[ReducedShard]:
        self.execution_counters._record_run()
        try:
            result = self._operation(tensor) if out is None else self._operation(tensor, out=out)
            return bind_execution_work(result, self.execution_info, self.execution_counters)
        except BaseException:
            self.execution_counters._record_failed()
            raise


# Backward-compatible name for callers that adopted the Task 5 executor.
CudaReducedShardExecutor = CompressedReduceScatterExecutor
