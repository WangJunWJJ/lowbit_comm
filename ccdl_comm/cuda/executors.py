"""CUDA executors that own pre-bound production communication operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo, FallbackRecord
from ccdl_comm.shard import ReducedShard
from ccdl_comm.work import CollectiveWork, bind_execution_work

from .workspace import CudaOutputLease


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
        self.last_fallback_record: FallbackRecord | None = None
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
                self.last_fallback_record = None
                self.last_execution_info = self._fused_execution_info
                result = output
            elif isinstance(execution, str):
                if not execution:
                    raise ValueError("fallback reason must not be empty")
                self._record_fallback(execution)
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
                    self.last_fallback_record = None
                    self.last_execution_info = self._fused_execution_info
                else:
                    self._record_fallback(execution.fallback_reason)
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

    def _record_fallback(self, reason: str) -> None:
        record = FallbackRecord(
            reason=reason,
            from_path=self.execution_info.fast_path,
            to_path="python_fallback",
        )
        self.last_fallback_record = record
        self.execution_counters._record_fallback(record)


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
        self._output_owner_token = getattr(operation, "output_owner_token", None)
        self._acquire_output = getattr(operation, "acquire_output", None)
        self.execution_info = execution_info
        self.execution_counters = ExecutionCounters()

    def acquire_output(self) -> CudaOutputLease:
        """Acquire one explicitly owned pooled ReducedShard output buffer."""

        if not callable(self._acquire_output):
            raise RuntimeError("ReducedShard output cache is disabled for this executor")
        return self._acquire_output()

    def run(self, tensor: object, *, out: object | None = None) -> CollectiveWork[ReducedShard]:
        self.execution_counters._record_run()
        output_lease = out if isinstance(out, CudaOutputLease) else None
        lease_marked = False
        try:
            if output_lease is not None:
                if self._output_owner_token is None:
                    raise RuntimeError("ReducedShard output cache is disabled for this executor")
                out = output_lease.mark_used(self._output_owner_token)
                lease_marked = True
            result = self._operation(tensor) if out is None else self._operation(tensor, out=out)
            work = bind_execution_work(result, self.execution_info, self.execution_counters)
            if output_lease is not None:
                retained_work = _LeaseRetainingWork(work, output_lease)
                output_lease.bind_work(self._output_owner_token, retained_work)
                return retained_work
            return work
        except BaseException:
            if output_lease is not None and lease_marked:
                output_lease.abort_use(self._output_owner_token)
            self.execution_counters._record_failed()
            raise


class _LeaseRetainingWork(CollectiveWork[ReducedShard]):
    """Keep an explicitly leased output alive while delegated work is active."""

    def __init__(self, delegate: CollectiveWork[ReducedShard], lease: CudaOutputLease) -> None:
        self._delegate = delegate
        self._lease = lease
        self._future: _LeaseRetainingFuture | None = None

    @property
    def resources(self) -> tuple[object, ...]:
        return (*tuple(getattr(self._delegate, "resources", ())), self._lease)

    @property
    def execution_info(self) -> ExecutionInfo | None:
        return self._delegate.execution_info

    @property
    def execution_counters(self) -> ExecutionCounters | None:
        return self._delegate.execution_counters

    def wait(self) -> ReducedShard:
        return self._delegate.wait()

    def query(self) -> bool:
        return self._delegate.query()

    def get_future(self) -> object | None:
        future = self._delegate.get_future()
        if future is None:
            return None
        if self._future is None:
            self._future = _LeaseRetainingFuture(future, self._lease)
        return self._future


class _LeaseRetainingFuture:
    """Forward future behavior while preserving the output-lease lifetime."""

    def __init__(self, delegate: object, lease: CudaOutputLease) -> None:
        self._delegate = delegate
        self._lease = lease

    def then(self, callback: Callable[[object], object]) -> object:
        result = self._delegate.then(callback)
        return _LeaseRetainingFuture(result, self._lease)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


# Backward-compatible name for callers that adopted the Task 5 executor.
CudaReducedShardExecutor = CompressedReduceScatterExecutor
