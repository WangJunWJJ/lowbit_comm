from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ccdl_comm import ExecutionCounters, ExecutionInfo
from ccdl_comm.cuda.executors import CudaAllReduceExecutor
from ccdl_comm.work import CompletionWork, ImmediateWork


INFO = ExecutionInfo(
    requested_strategy="all_gather",
    executed_strategy="all_gather",
    backend="cuda",
    fallback_used=False,
    fallback_reason=None,
    stage_names=(),
    original_bytes=2048,
    compressed_bytes=1024,
    compression_ratio=2.0,
    workspace_cache_hit=False,
    async_capable=True,
    fast_path="cuda_all_gather",
)


class ReadyHandle:
    def wait(self) -> None:
        return None

    def is_completed(self) -> bool:
        return True


def test_query_is_observational_and_callback_runs_once() -> None:
    callbacks = []
    counters = ExecutionCounters()
    work = CompletionWork(
        result=None,
        handle=ReadyHandle(),
        complete=lambda: callbacks.append("done") or 7,
        execution_info=INFO,
        execution_counters=counters,
    )

    assert work.query() is False
    assert callbacks == []
    assert work.wait() == 7
    assert work.wait() == 7
    assert callbacks == ["done"]
    assert work.execution_info is INFO
    assert counters.snapshot().query_calls == 1
    assert counters.snapshot().wait_calls == 2
    assert counters.snapshot().completed_runs == 1


def test_execution_info_and_counter_snapshot_are_read_only() -> None:
    counters = ExecutionCounters()
    snapshot = counters.snapshot()
    work = ImmediateWork(3, execution_info=INFO, execution_counters=counters)

    with pytest.raises((AttributeError, FrozenInstanceError)):
        work.execution_info = None
    with pytest.raises((AttributeError, FrozenInstanceError)):
        snapshot.wait_calls = 9


def test_cuda_executor_attaches_metadata_and_shared_counters_to_each_work() -> None:
    executor = CudaAllReduceExecutor(lambda tensor: tensor + 1, INFO)

    first = executor.run(1)
    second = executor.run(2)

    assert first.execution_info is INFO
    assert second.execution_info is INFO
    assert first.execution_counters is executor.execution_counters
    assert second.execution_counters is executor.execution_counters
    assert first.wait() == 2
    assert second.wait() == 3
    assert executor.execution_counters.snapshot().run_calls == 2
    assert executor.execution_counters.snapshot().completed_runs == 2


def test_cuda_executor_counts_launch_and_async_completion_failures_once() -> None:
    launch_error = RuntimeError("launch failed")
    launch_executor = CudaAllReduceExecutor(
        lambda tensor: (_ for _ in ()).throw(launch_error),
        INFO,
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        launch_executor.run(1)
    assert launch_executor.execution_counters.snapshot().failed_runs == 1

    completion_error = RuntimeError("completion failed")
    completion_executor = CudaAllReduceExecutor(
        lambda tensor: CompletionWork(
            None,
            complete=lambda: (_ for _ in ()).throw(completion_error),
        ),
        INFO,
    )
    work = completion_executor.run(1)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="completion failed"):
            work.wait()
    assert completion_executor.execution_counters.snapshot().failed_runs == 1


def test_cuda_executor_wraps_duck_typed_work_without_losing_future() -> None:
    future = object()

    class ForeignWork:
        def wait(self):
            return "done"

        def query(self):
            return True

        def get_future(self):
            return future

    executor = CudaAllReduceExecutor(lambda tensor: ForeignWork(), INFO)
    work = executor.run("tensor")

    assert work.execution_info is INFO
    assert work.wait() == "done"
    assert work.query() is True
    assert work.get_future() is future


def test_cuda_executor_wraps_pytorch_style_work_with_is_completed() -> None:
    calls = []

    class TorchWork:
        def wait(self):
            calls.append("wait")
            return "done"

        def is_completed(self):
            calls.append("query")
            return False

    work = CudaAllReduceExecutor(lambda tensor: TorchWork(), INFO).run("tensor")

    assert work.query() is False
    assert work.wait() == "done"
    assert calls == ["query", "wait"]


def test_bound_foreign_work_caches_result_and_error() -> None:
    calls = []

    class SuccessfulWork:
        def wait(self):
            calls.append("success")
            return 9

        def query(self):
            return False

    successful = CudaAllReduceExecutor(lambda tensor: SuccessfulWork(), INFO).run(1)
    assert successful.wait() == 9
    assert successful.wait() == 9
    assert calls == ["success"]

    error = RuntimeError("foreign failure")

    class FailingWork:
        def wait(self):
            calls.append("failure")
            raise error

        def query(self):
            return False

    failed = CudaAllReduceExecutor(lambda tensor: FailingWork(), INFO).run(1)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            failed.wait()
        assert raised.value is error
    assert calls == ["success", "failure"]


def test_binding_existing_work_records_terminal_state_exactly_once() -> None:
    counters = ExecutionCounters()
    executor = CudaAllReduceExecutor(
        lambda tensor: ImmediateWork(
            tensor,
            execution_info=INFO,
            execution_counters=counters,
        ),
        INFO,
    )
    executor.execution_counters = counters

    assert executor.run(3).wait() == 3
    assert counters.snapshot().completed_runs == 1

    completed = CompletionWork(7)
    assert completed.wait() == 7
    late_bound_executor = CudaAllReduceExecutor(lambda tensor: completed, INFO)
    rebound = late_bound_executor.run(1)
    assert rebound.wait() == 7
    assert late_bound_executor.execution_counters.snapshot().completed_runs == 1


def test_completion_work_allows_core_to_replace_static_execution_info() -> None:
    replacement = ExecutionInfo(
        requested_strategy="topology",
        executed_strategy="all_gather",
        backend="cuda",
        fallback_used=True,
        fallback_reason="explicit fallback from topology to all_gather",
        stage_names=(),
        original_bytes=2048,
        compressed_bytes=1024,
        compression_ratio=2.0,
        workspace_cache_hit=False,
        async_capable=True,
        fast_path="cuda_all_gather",
    )
    operation_work = CompletionWork(5, execution_info=INFO)
    executor = CudaAllReduceExecutor(lambda tensor: operation_work, replacement)

    work = executor.run(1)

    assert work.execution_info is replacement

    counters = ExecutionCounters()
    immediate = ImmediateWork(
        6,
        execution_info=INFO,
        execution_counters=counters,
    )
    immediate_executor = CudaAllReduceExecutor(lambda tensor: immediate, replacement)
    immediate_executor.execution_counters = counters

    rebound_immediate = immediate_executor.run(1)

    assert rebound_immediate.execution_info is replacement
    assert counters.snapshot().completed_runs == 1
