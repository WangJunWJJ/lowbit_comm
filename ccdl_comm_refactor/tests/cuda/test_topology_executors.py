from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import weakref

import pytest


class _FakeWork:
    def __init__(self, *, result: object, completion: object, resources: tuple[object, ...]) -> None:
        self.result = result
        self.completion = completion
        self.resources = resources

    def query(self) -> bool:
        return self.completion.query()  # type: ignore[no-any-return, union-attr]

    def wait(self) -> object:
        self.completion.wait()  # type: ignore[union-attr]
        return self.result


class _FakeCompletionManager:
    def create_work(
        self,
        *,
        result: object,
        completion: object,
        resources: tuple[object, ...],
    ) -> _FakeWork:
        return _FakeWork(result=result, completion=completion, resources=resources)


class _FailingCompletionManager:
    def __init__(self) -> None:
        self.calls = 0

    def create_work(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        raise RuntimeError("create_work failed")


class _FakeCompletion:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.ready = False

    def query(self) -> bool:
        self.calls.append("query")
        return self.ready

    def wait(self) -> None:
        self.calls.append("cpu_wait")
        self.ready = True


class _FakeSubmissionContext:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.ready = False

    def query(self) -> bool:
        self.calls.append("context_query")
        return self.ready


class _FakeDependency:
    def query(self) -> bool:
        return False


class _FakeWorkspaceSession:
    def __init__(self) -> None:
        self.buffers = {"workspace": object()}
        self.release_calls: list[object] = []
        self.abort_calls = 0

    def release(self, *, completion: object) -> None:
        self.release_calls.append(completion)

    def abort(self) -> None:
        self.abort_calls += 1


class _ReleaseOnlyWorkspaceSession:
    def __init__(self, *, fail_release: bool = False) -> None:
        self.buffers = {"workspace": object()}
        self.release_calls: list[object] = []
        self.fail_release = fail_release

    def release(self, *, completion: object) -> None:
        self.release_calls.append(completion)
        if self.fail_release:
            raise RuntimeError("workspace release failed")


class _FakeRingRuntime:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.handles: list[object] = []
        self.completion = _FakeCompletion(self.calls)
        self.context = _FakeSubmissionContext(self.calls)
        self.record_dependencies: tuple[object, ...] = ()

    def create_submission_context(self, tensor: object) -> _FakeSubmissionContext:
        self.calls.append(("create_submission_context", tensor))
        return self.context

    def wait_for_producer(self, tensor: object, *, context: object) -> None:
        assert context is self.context
        self.calls.append(("producer_wait", tensor))

    def quant_pack(
        self, tensor: object, chunk: object, workspace: object, *, context: object
    ) -> object:
        assert context is self.context
        payload = ("packed", chunk)
        self.calls.append(("quant_pack", chunk, workspace))
        return payload

    def send_recv(
        self,
        payload: object,
        *,
        send_peer: int,
        recv_peer: int,
        recv_chunk: object,
        workspace: object,
        context: object,
    ) -> tuple[object, object]:
        assert context is self.context
        handle = _FakeDependency()
        self.handles.append(handle)
        self.calls.append(("send_recv", send_peer, recv_peer, recv_chunk, workspace))
        return ("received", recv_chunk), handle

    def fused_reduce(
        self,
        tensor: object,
        received: object,
        chunk: object,
        contributors: tuple[int, ...],
        workspace: object,
        *,
        context: object,
        dependency: object,
    ) -> object:
        assert context is self.context
        assert dependency in self.handles
        self.calls.append(("fused_reduce", chunk, contributors, workspace))
        return object()

    def record_completion(
        self, *, context: object, dependencies: tuple[object, ...]
    ) -> _FakeCompletion:
        assert context is self.context
        self.record_dependencies = dependencies
        self.calls.append("record_completion")
        return self.completion

    def wait(self) -> None:
        raise AssertionError("run() must not call runtime.wait()")

    def synchronize(self) -> None:
        raise AssertionError("run() must not call runtime.synchronize()")


class _DependencyRuntime(_FakeRingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.context = _FakeSubmissionContext(self.calls)
        self.communication = _FakeDependency()
        self.reduction = object()

    def create_submission_context(self, tensor: object) -> object:
        return self.context

    def wait_for_producer(self, tensor: object, *, context: object) -> None:
        assert context is self.context

    def quant_pack(
        self, tensor: object, chunk: object, workspace: object, *, context: object
    ) -> object:
        assert context is self.context
        return object()

    def send_recv(
        self, payload: object, *, context: object, **kwargs: object
    ) -> tuple[object, object]:
        assert context is self.context
        return object(), self.communication

    def fused_reduce(
        self,
        tensor: object,
        received: object,
        chunk: object,
        contributors: tuple[int, ...],
        workspace: object,
        *,
        context: object,
        dependency: object,
    ) -> object:
        assert context is self.context
        assert dependency is self.communication
        return self.reduction

    def record_completion(
        self, *, context: object, dependencies: tuple[object, ...]
    ) -> _FakeCompletion:
        assert context is self.context
        assert self.communication in dependencies
        assert self.reduction in dependencies
        return self.completion


class _FakeTreeRuntime:
    def __init__(self, *, fail_on_quant: int | None = None) -> None:
        self.calls: list[object] = []
        self.handles: list[object] = []
        self.completion = _FakeCompletion(self.calls)
        self.context = _FakeSubmissionContext(self.calls)
        self.record_dependencies: tuple[object, ...] = ()
        self._quant_count = 0
        self._fail_on_quant = fail_on_quant

    def create_submission_context(self, tensor: object) -> _FakeSubmissionContext:
        self.calls.append(("create_submission_context", tensor))
        return self.context

    def wait_for_producer(self, tensor: object, *, context: object) -> None:
        assert context is self.context
        self.calls.append(("producer_wait", tensor))

    def quant_pack(
        self, tensor: object, edge: object, workspace: object, *, context: object
    ) -> object:
        assert context is self.context
        self._quant_count += 1
        if self._quant_count == self._fail_on_quant:
            raise RuntimeError("quant submission failed")
        payload = ("packed", edge)
        self.calls.append(("quant_pack", edge, workspace))
        return payload

    def send(
        self,
        payload: object,
        *,
        peer: int,
        edge: object,
        workspace: object,
        context: object,
    ) -> object:
        assert context is self.context
        handle = _FakeDependency()
        self.handles.append(handle)
        self.calls.append(("send", peer, edge, workspace))
        return handle

    def receive(
        self,
        *,
        peer: int,
        edge: object,
        workspace: object,
        context: object,
    ) -> tuple[object, object]:
        assert context is self.context
        payload = ("received", edge)
        handle = _FakeDependency()
        self.handles.append(handle)
        self.calls.append(("receive", peer, edge, workspace))
        return payload, handle

    def fused_reduce(
        self,
        tensor: object,
        received: object,
        edge: object,
        workspace: object,
        *,
        context: object,
        dependency: object,
    ) -> object:
        assert context is self.context
        assert dependency in self.handles
        self.calls.append(("fused_reduce", edge, workspace))
        return object()

    def apply_broadcast(
        self,
        tensor: object,
        received: object,
        edge: object,
        workspace: object,
        *,
        context: object,
        dependency: object,
    ) -> object:
        assert context is self.context
        assert dependency in self.handles
        self.calls.append(("apply_broadcast", edge, workspace))
        return object()

    def record_completion(
        self, *, context: object, dependencies: tuple[object, ...]
    ) -> _FakeCompletion:
        assert context is self.context
        self.record_dependencies = dependencies
        self.calls.append("record_completion")
        return self.completion

    def wait(self) -> None:
        raise AssertionError("run() must not call runtime.wait()")

    def synchronize(self) -> None:
        raise AssertionError("run() must not call runtime.synchronize()")

    def registry(self) -> None:
        raise AssertionError("run() must not call Registry")

    def planner(self) -> None:
        raise AssertionError("run() must not call planner")

    def select_strategy(self) -> None:
        raise AssertionError("run() must not select a strategy")


@pytest.mark.parametrize("world_size", (3, 5, 8))
def test_pipelined_ring_schedule_has_one_deterministic_step_per_remote_chunk(world_size: int) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    plan = compile_chunk_plan(original_numel=world_size * 7 + 1, world_size=world_size)

    for rank in range(world_size):
        schedule = compile_pipelined_ring_schedule(chunk_plan=plan, rank=rank)

        assert schedule.rank == rank
        assert schedule.chunk_plan is plan
        assert len(schedule.steps) == world_size - 1
        assert tuple(step.step_index for step in schedule.steps) == tuple(range(world_size - 1))
        assert {step.send_peer for step in schedule.steps} == {(rank + 1) % world_size}
        assert {step.recv_peer for step in schedule.steps} == {(rank - 1) % world_size}
        assert tuple(step.send_chunk_owner for step in schedule.steps) == tuple(
            (rank - step_index - 1) % world_size for step_index in range(world_size - 1)
        )
        assert tuple(step.recv_chunk_owner for step in schedule.steps) == tuple(
            (rank - step_index - 2) % world_size for step_index in range(world_size - 1)
        )
        assert set(step.recv_chunk_owner for step in schedule.steps) == set(range(world_size)) - {
            (rank - 1) % world_size
        }
        assert schedule.steps[-1].recv_chunk_owner == rank
        assert all(step.send_chunk == plan.chunk_for_rank(step.send_chunk_owner) for step in schedule.steps)
        assert all(step.recv_chunk == plan.chunk_for_rank(step.recv_chunk_owner) for step in schedule.steps)


@pytest.mark.parametrize("world_size", (3, 5, 8))
def test_pipelined_ring_schedule_tracks_aggregate_provenance_through_every_rank(world_size: int) -> None:
    """Simulate the scheduled messages without reproducing the compiler's formulas."""

    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    plan = compile_chunk_plan(original_numel=world_size * 5, world_size=world_size)
    schedules = tuple(
        compile_pipelined_ring_schedule(chunk_plan=plan, rank=rank)
        for rank in range(world_size)
    )
    chunks_by_rank = [
        {owner: (rank,) for owner in range(world_size)}
        for rank in range(world_size)
    ]

    for step_index in range(world_size - 1):
        messages = {
            (rank, schedule.steps[step_index].send_peer): chunks_by_rank[rank][
                schedule.steps[step_index].send_chunk_owner
            ]
            for rank, schedule in enumerate(schedules)
        }
        for rank, schedule in enumerate(schedules):
            step = schedule.steps[step_index]
            received = messages[(step.recv_peer, rank)]
            existing = chunks_by_rank[rank][step.recv_chunk_owner]

            assert step.received_contributors == received
            assert not set(received).intersection(existing)
            chunks_by_rank[rank][step.recv_chunk_owner] = received + existing

    expected_contributors = set(range(world_size))
    for rank in range(world_size):
        assert set(chunks_by_rank[rank][rank]) == expected_contributors


def test_pipelined_ring_executor_submits_every_step_in_order_without_cpu_wait() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    plan = compile_chunk_plan(original_numel=15, world_size=3)
    schedule = compile_pipelined_ring_schedule(chunk_plan=plan, rank=1)
    runtime = _FakeRingRuntime()
    session = _FakeWorkspaceSession()
    tensor = object()
    executor = PipelinedRingExecutor(
        schedule=schedule,
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    work = executor.run(tensor)

    assert runtime.calls == [
        ("create_submission_context", tensor),
        ("producer_wait", tensor),
        ("quant_pack", schedule.steps[0].send_chunk, session),
        (
            "send_recv",
            schedule.steps[0].send_peer,
            schedule.steps[0].recv_peer,
            schedule.steps[0].recv_chunk,
            session,
        ),
        (
            "fused_reduce",
            schedule.steps[0].recv_chunk,
            schedule.steps[0].received_contributors,
            session,
        ),
        ("quant_pack", schedule.steps[1].send_chunk, session),
        (
            "send_recv",
            schedule.steps[1].send_peer,
            schedule.steps[1].recv_peer,
            schedule.steps[1].recv_chunk,
            session,
        ),
        (
            "fused_reduce",
            schedule.steps[1].recv_chunk,
            schedule.steps[1].received_contributors,
            session,
        ),
        "record_completion",
    ]
    assert work.result is tensor
    assert work.completion is runtime.completion
    assert session in work.resources
    assert all(handle in work.resources for handle in runtime.handles)
    assert all(handle in runtime.record_dependencies for handle in runtime.handles)
    assert session.release_calls == [runtime.completion]
    assert session.abort_calls == 0
    assert "cpu_wait" not in runtime.calls


def test_ring_step_contributor_count_matches_submission_depth() -> None:
    from ccdl_comm.cuda.transports import ChunkRange, RingReduceScatterStep

    with pytest.raises(ValueError, match=r"step_index \+ 1"):
        RingReduceScatterStep(
            step_index=1,
            send_peer=1,
            recv_peer=2,
            send_chunk_owner=0,
            recv_chunk_owner=2,
            received_contributors=(0,),
            send_chunk=ChunkRange(0, 1),
            recv_chunk=ChunkRange(1, 2),
        )


def test_ring_executor_passes_explicit_context_and_dependencies() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    runtime = _DependencyRuntime()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: _FakeWorkspaceSession(),
        completion_manager=_FakeCompletionManager(),
    )

    executor.run(object())


@pytest.mark.parametrize(
    "dependency",
    (
        pytest.param(_FakeDependency(), id="query"),
        pytest.param(
            type("IsCompletedDependency", (), {"is_completed": lambda self: False})(),
            id="is-completed",
        ),
    ),
)
def test_ring_executor_accepts_only_nonblocking_p2p_dependencies(dependency: object) -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class Runtime(_FakeRingRuntime):
        def send_recv(self, *args: object, **kwargs: object) -> tuple[object, object]:
            del args, kwargs
            self.handles.append(dependency)
            return object(), dependency

    runtime = Runtime()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: _ReleaseOnlyWorkspaceSession(),
        completion_manager=_FakeCompletionManager(),
    )

    executor.run(object())


def test_ring_executor_rejects_blocking_only_p2p_dependency_contract() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class BlockingDependency:
        def wait(self) -> None:
            raise AssertionError("blocking dependency must never be waited")

    class Runtime(_FakeRingRuntime):
        def send_recv(self, *args: object, **kwargs: object) -> tuple[object, object]:
            del args, kwargs
            dependency = BlockingDependency()
            self.handles.append(dependency)
            return object(), dependency

    runtime = Runtime()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: _ReleaseOnlyWorkspaceSession(),
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(TypeError, match="P2P dependency.*query.*is_completed"):
        executor.run(object())


def test_ring_submission_failure_aborts_workspace_once() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class FailingRuntime(_FakeRingRuntime):
        def send_recv(self, *args: object, **kwargs: object) -> tuple[object, object]:
            raise RuntimeError("communication submission failed")

    plan = compile_chunk_plan(original_numel=9, world_size=3)
    runtime = FailingRuntime()
    session = _FakeWorkspaceSession()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(chunk_plan=plan, rank=0),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="communication submission failed"):
        executor.run(object())

    assert session.abort_calls == 0
    assert session.release_calls == []
    assert executor.pending_submission_count == 1
    runtime.completion.ready = True
    executor.reap_pending()
    assert session.abort_calls == 1
    assert executor.pending_submission_count == 0
    assert "cpu_wait" not in runtime.calls


def test_workspace_factory_failure_transfers_no_ownership() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    runtime = _FakeRingRuntime()

    def fail_factory(tensor: object) -> _FakeWorkspaceSession:
        del tensor
        raise RuntimeError("workspace factory failed")

    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=fail_factory,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="workspace factory failed"):
        executor.run(object())

    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] == "create_submission_context"
    assert executor.pending_submission_count == 0


def test_record_failure_is_not_retried_or_masked_and_quarantines_resources() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class RecordFailRuntime(_FakeRingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.record_calls = 0

        def record_completion(self, **kwargs: object) -> _FakeCompletion:
            del kwargs
            self.record_calls += 1
            raise RuntimeError("record completion failed")

    runtime = RecordFailRuntime()
    session = _FakeWorkspaceSession()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="record completion failed"):
        executor.run(object())

    assert runtime.record_calls == 1
    assert session.abort_calls == 0
    assert executor.pending_submission_count == 1
    runtime.context.ready = True
    executor.reap_pending()
    assert session.abort_calls == 1
    assert executor.pending_submission_count == 0


def test_release_only_cleanup_preserves_submission_error_when_record_fails() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class SubmitAndRecordFailRuntime(_FakeRingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.record_calls = 0

        def send_recv(self, *args: object, **kwargs: object) -> tuple[object, object]:
            raise RuntimeError("communication submission failed")

        def record_completion(self, **kwargs: object) -> _FakeCompletion:
            del kwargs
            self.record_calls += 1
            raise RuntimeError("record completion failed")

    runtime = SubmitAndRecordFailRuntime()
    session = _ReleaseOnlyWorkspaceSession()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="communication submission failed"):
        executor.run(object())

    assert runtime.record_calls == 1
    assert session.release_calls == []
    assert executor.pending_submission_count == 1
    runtime.context.ready = True
    executor.reap_pending()
    assert session.release_calls == [runtime.context]
    assert runtime.record_calls == 1
    assert executor.pending_submission_count == 0


def test_real_release_only_workspace_session_is_reaped_without_cpu_wait() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule
    from ccdl_comm.cuda.workspace import (
        CudaShardWorkspaceProvider,
        CudaWorkspacePool,
        WorkspaceKey,
    )

    class RecordFailRuntime(_FakeRingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.record_calls = 0

        def record_completion(self, **kwargs: object) -> _FakeCompletion:
            del kwargs
            self.record_calls += 1
            raise RuntimeError("record completion failed")

    pool = CudaWorkspacePool(allocator=lambda key, stream: object())
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="test",
        collective="reduce_scatter",
        strategy="tree",
        device="cuda",
    )
    session = provider.begin(stream=None)
    session._acquire(  # noqa: SLF001 - exercise the real release-only session shape
        WorkspaceKey(
            backend="test",
            collective="reduce_scatter",
            strategy="tree",
            shape_class=(8,),
            dtype="uint8",
            world_size=2,
            bit=8,
            group_size=1,
            chunk_config=(0,),
            workspace_kind="send",
        )
    )
    runtime = RecordFailRuntime()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="record completion failed"):
        executor.run(object())

    assert pool.stats.in_flight_bytes == 8
    runtime.context.ready = True
    executor.reap_pending()
    assert pool.stats.in_flight_bytes == 0
    assert runtime.record_calls == 1
    assert "cpu_wait" not in runtime.calls


def test_create_work_failure_establishes_pending_owner_before_workspace_release() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    calls: list[str] = []

    class Session(_ReleaseOnlyWorkspaceSession):
        def release(self, *, completion: object) -> None:
            calls.append("release")
            super().release(completion=completion)

    class Manager(_FailingCompletionManager):
        def create_work(self, **kwargs: object) -> object:
            calls.append("create_work")
            return super().create_work(**kwargs)

    runtime = _FakeRingRuntime()
    session = Session()
    manager = Manager()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=manager,
    )

    with pytest.raises(RuntimeError, match="create_work failed"):
        executor.run(object())

    assert manager.calls == 1
    assert calls == ["create_work"]
    assert session.release_calls == []
    assert executor.pending_submission_count == 1
    runtime.completion.ready = True
    executor.reap_pending()
    assert calls == ["create_work", "release"]
    assert session.release_calls == [runtime.completion]
    assert executor.pending_submission_count == 0


def test_release_failure_keeps_completed_work_resources_pending_until_query() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    runtime = _FakeRingRuntime()
    session = _ReleaseOnlyWorkspaceSession(fail_release=True)
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="workspace release failed"):
        executor.run(object())

    assert executor.pending_submission_count == 1
    runtime.completion.ready = True
    session.fail_release = False
    executor.reap_pending()
    assert session.release_calls == [runtime.completion, runtime.completion]
    assert executor.pending_submission_count == 0


def test_late_submission_failure_retains_async_resources_until_recorded_completion() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule

    class Retained:
        def query(self) -> bool:
            return False

    class Runtime(_FakeRingRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.quant_calls = 0
            self.payload_ref: weakref.ReferenceType[Retained] | None = None
            self.handle_ref: weakref.ReferenceType[Retained] | None = None

        def quant_pack(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            self.quant_calls += 1
            if self.quant_calls == 2:
                raise RuntimeError("late quant submission failed")
            payload = Retained()
            self.payload_ref = weakref.ref(payload)
            return payload

        def send_recv(self, *args: object, **kwargs: object) -> tuple[object, object]:
            del args, kwargs
            handle = Retained()
            self.handle_ref = weakref.ref(handle)
            return Retained(), handle

        def fused_reduce(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return Retained()

    runtime = Runtime()
    session = _FakeWorkspaceSession()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=6, world_size=3), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="late quant submission failed"):
        executor.run(object())

    assert runtime.payload_ref is not None and runtime.payload_ref() is not None
    assert runtime.handle_ref is not None and runtime.handle_ref() is not None
    assert session.abort_calls == 0
    assert executor.pending_submission_count == 1

    runtime.context.ready = True
    executor.reap_pending()
    assert session.abort_calls == 0
    assert executor.pending_submission_count == 1

    runtime.completion.ready = True
    executor.reap_pending()
    assert session.abort_calls == 1
    assert executor.pending_submission_count == 0
    executor.reap_pending()
    assert session.abort_calls == 1


def test_partial_real_session_release_reclaims_every_captured_lease() -> None:
    from ccdl_comm.cuda.transports import PipelinedRingExecutor, compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule
    from ccdl_comm.cuda.workspace import (
        CudaShardWorkspaceProvider,
        CudaWorkspacePool,
        WorkspaceKey,
    )

    class FailSecondReleasePool(CudaWorkspacePool):
        def __init__(self) -> None:
            super().__init__(allocator=lambda key, stream: object())
            self.release_calls = 0

        def _release(self, record: object, completion: object) -> None:
            self.release_calls += 1
            if self.release_calls == 2:
                raise RuntimeError("second lease release failed")
            super()._release(record, completion)  # type: ignore[arg-type]

    pool = FailSecondReleasePool()
    provider = CudaShardWorkspaceProvider(
        pool,
        backend="test",
        collective="reduce_scatter",
        strategy="ring",
        device="cuda",
    )
    session = provider.begin(stream=None)
    for index in range(3):
        session._acquire(  # noqa: SLF001 - exercise real partial-release semantics
            WorkspaceKey(
                backend="test",
                collective="reduce_scatter",
                strategy="ring",
                shape_class=(8,),
                dtype="uint8",
                world_size=2,
                bit=8,
                group_size=1,
                chunk_config=(index,),
                workspace_kind=f"lease-{index}",
            )
        )
    captured = session.leases
    runtime = _FakeRingRuntime()
    executor = PipelinedRingExecutor(
        schedule=compile_pipelined_ring_schedule(
            chunk_plan=compile_chunk_plan(original_numel=2, world_size=2), rank=0
        ),
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="second lease release failed"):
        executor.run(object())

    assert executor.pending_submission_count == 1
    runtime.completion.ready = True
    executor.reap_pending()

    assert all(lease.released for lease in captured)
    assert pool.stats.in_flight_bytes == 0
    assert executor.pending_submission_count == 0


@pytest.mark.parametrize("world_size", (3, 5, 8))
def test_tree_schedule_covers_each_rank_once_with_a_connected_acyclic_topology(world_size: int) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
    from ccdl_comm.cuda.transports.tree import compile_tree_schedule

    plan = compile_chunk_plan(original_numel=world_size * 5, world_size=world_size)

    for root in range(world_size):
        schedule = compile_tree_schedule(chunk_plan=plan, rank=root, root=root)

        assert schedule.parent is None
        assert schedule.local_chunk == plan.chunk_for_rank(root)
        assert len(schedule.reduce_edges) == world_size - 1
        assert len(schedule.broadcast_edges) == world_size - 1
        assert schedule.broadcast_edges == tuple(reversed(schedule.reduce_edges))
        assert {edge.child_rank for edge in schedule.reduce_edges} == set(range(world_size)) - {root}
        assert all(edge.parent_rank != edge.child_rank for edge in schedule.reduce_edges)

        parents = {
            edge.child_rank: edge.parent_rank
            for edge in schedule.reduce_edges
        }
        assert len(parents) == world_size - 1
        for child in parents:
            current = child
            visited = set()
            while current != root:
                assert current not in visited
                visited.add(current)
                current = parents[current]

        for rank in range(world_size):
            local = compile_tree_schedule(chunk_plan=plan, rank=rank, root=root)
            assert local.parent == parents.get(rank)
            assert local.children == tuple(
                edge.child_rank for edge in schedule.reduce_edges if edge.parent_rank == rank
            )


def test_tree_executor_orders_reduce_before_reverse_broadcast_for_non_power_of_two_world() -> None:
    from ccdl_comm.cuda.transports import TreeExecutor, compile_chunk_plan, compile_tree_schedule

    schedule = compile_tree_schedule(
        chunk_plan=compile_chunk_plan(original_numel=25, world_size=5),
        rank=1,
        root=0,
    )
    edge_4_to_1, edge_3_to_1, _edge_2_to_0, edge_1_to_0 = schedule.reduce_edges
    runtime = _FakeTreeRuntime()
    session = _FakeWorkspaceSession()
    tensor = object()
    executor = TreeExecutor(
        schedule=schedule,
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    work = executor.run(tensor)

    assert runtime.calls == [
        ("create_submission_context", tensor),
        ("producer_wait", tensor),
        ("receive", 4, edge_4_to_1, session),
        ("fused_reduce", edge_4_to_1, session),
        ("receive", 3, edge_3_to_1, session),
        ("fused_reduce", edge_3_to_1, session),
        ("quant_pack", edge_1_to_0, session),
        ("send", 0, edge_1_to_0, session),
        ("receive", 0, edge_1_to_0, session),
        ("apply_broadcast", edge_1_to_0, session),
        ("quant_pack", edge_3_to_1, session),
        ("send", 3, edge_3_to_1, session),
        ("quant_pack", edge_4_to_1, session),
        ("send", 4, edge_4_to_1, session),
        "record_completion",
    ]
    assert work.result is tensor
    assert work.query() is False
    assert session in work.resources
    assert all(handle in work.resources for handle in runtime.handles)
    assert all(handle in runtime.record_dependencies for handle in runtime.handles)
    assert session.release_calls == [runtime.completion]
    assert session.abort_calls == 0
    assert "cpu_wait" not in runtime.calls


def test_five_rank_tree_matches_every_send_receive_without_blocking() -> None:
    from ccdl_comm.cuda.transports import TreeExecutor, compile_chunk_plan, compile_tree_schedule

    class Handle:
        def __init__(self) -> None:
            self.ready = False

        def query(self) -> bool:
            return self.ready

    class Context:
        def __init__(self) -> None:
            self.handles: list[Handle] = []

        def query(self) -> bool:
            return all(handle.query() for handle in self.handles)

    class Network:
        def __init__(self) -> None:
            self.posts: dict[tuple[object, ...], dict[str, Handle]] = {}

        def post(self, kind: str, key: tuple[object, ...]) -> Handle:
            handle = Handle()
            pair = self.posts.setdefault(key, {})
            assert kind not in pair
            pair[kind] = handle
            if set(pair) == {"send", "receive"}:
                pair["send"].ready = True
                pair["receive"].ready = True
            return handle

    class Runtime:
        def __init__(self, rank: int, network: Network) -> None:
            self.rank = rank
            self.network = network
            self.context = Context()
            self.phases: list[str] = []

        def create_submission_context(self, tensor: object) -> Context:
            del tensor
            return self.context

        def wait_for_producer(self, tensor: object, *, context: object) -> None:
            del tensor
            assert context is self.context

        def quant_pack(
            self, tensor: object, edge: object, workspace: object, *, context: object
        ) -> object:
            del tensor, workspace
            assert context is self.context
            return edge

        def send(
            self,
            payload: object,
            *,
            peer: int,
            edge: object,
            workspace: object,
            context: object,
        ) -> Handle:
            del payload, workspace
            assert context is self.context
            key = (
                ("reduce", edge.child_rank, edge.parent_rank)
                if self.rank == edge.child_rank
                else ("broadcast", edge.parent_rank, edge.child_rank)
            )
            self.phases.append(key[0])
            handle = self.network.post("send", key)
            self.context.handles.append(handle)
            return handle

        def receive(
            self,
            *,
            peer: int,
            edge: object,
            workspace: object,
            context: object,
        ) -> tuple[object, Handle]:
            del peer, workspace
            assert context is self.context
            key = (
                ("reduce", edge.child_rank, edge.parent_rank)
                if self.rank == edge.parent_rank
                else ("broadcast", edge.parent_rank, edge.child_rank)
            )
            self.phases.append(key[0])
            handle = self.network.post("receive", key)
            self.context.handles.append(handle)
            return edge, handle

        def fused_reduce(self, *args: object, context: object, dependency: object) -> object:
            del args
            assert context is self.context
            assert dependency in self.context.handles
            return dependency

        def apply_broadcast(self, *args: object, context: object, dependency: object) -> object:
            del args
            assert context is self.context
            assert dependency in self.context.handles
            return dependency

        def record_completion(
            self, *, context: object, dependencies: tuple[object, ...]
        ) -> Context:
            assert context is self.context
            assert all(handle in dependencies for handle in self.context.handles)
            return self.context

    plan = compile_chunk_plan(original_numel=25, world_size=5)
    network = Network()
    runtimes = [Runtime(rank, network) for rank in range(5)]
    works = []
    for rank, runtime in enumerate(runtimes):
        works.append(
            TreeExecutor(
                schedule=compile_tree_schedule(chunk_plan=plan, rank=rank, root=0),
                runtime=runtime,
                workspace_session_factory=lambda _tensor: _ReleaseOnlyWorkspaceSession(),
                completion_manager=_FakeCompletionManager(),
            ).run(object())
        )
        if rank == 0:
            assert len(network.posts) == 4
            assert any(set(pair) != {"send", "receive"} for pair in network.posts.values())
            assert works[0].query() is False

    assert len(network.posts) == 2 * (plan.world_size - 1)
    assert all(set(pair) == {"send", "receive"} for pair in network.posts.values())
    assert all(work.query() for work in works)
    assert all(
        phases == sorted(phases, key={"reduce": 0, "broadcast": 1}.__getitem__)
        for phases in (runtime.phases for runtime in runtimes)
    )


def test_tree_submission_failure_aborts_workspace_exactly_once() -> None:
    from ccdl_comm.cuda.transports import TreeExecutor, compile_chunk_plan, compile_tree_schedule

    schedule = compile_tree_schedule(
        chunk_plan=compile_chunk_plan(original_numel=25, world_size=5),
        rank=1,
        root=0,
    )
    runtime = _FakeTreeRuntime(fail_on_quant=1)
    session = _FakeWorkspaceSession()
    executor = TreeExecutor(
        schedule=schedule,
        runtime=runtime,
        workspace_session_factory=lambda _tensor: session,
        completion_manager=_FakeCompletionManager(),
    )

    with pytest.raises(RuntimeError, match="quant submission failed"):
        executor.run(object())

    assert session.abort_calls == 0
    assert session.release_calls == []
    assert "record_completion" in runtime.calls
    assert executor.pending_submission_count == 1
    runtime.completion.ready = True
    executor.reap_pending()
    assert session.abort_calls == 1
    assert executor.pending_submission_count == 0
    assert "cpu_wait" not in runtime.calls


def test_topology_schedule_metadata_is_immutable_and_imports_without_torch() -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule
    from ccdl_comm.cuda.transports.tree import compile_tree_schedule

    plan = compile_chunk_plan(original_numel=10, world_size=3)
    ring = compile_pipelined_ring_schedule(chunk_plan=plan, rank=0)
    tree = compile_tree_schedule(chunk_plan=plan, rank=1, root=0)

    with pytest.raises(FrozenInstanceError):
        ring.rank = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tree.parent = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ring.steps[0].send_peer = 2  # type: ignore[misc]

    source_root = Path(__file__).resolve().parents[2]
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ccdl_comm.cuda.transports.pipelined_ring; "
            "import ccdl_comm.cuda.transports.tree; assert 'torch' not in sys.modules",
        ],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr


@pytest.mark.parametrize(
    ("constructor", "kwargs", "exception", "message"),
    (
        (
            "ring",
            {"step_index": False},
            TypeError,
            "step_index",
        ),
        (
            "ring",
            {"send_peer": -1},
            ValueError,
            "send_peer",
        ),
        (
            "ring",
            {"received_contributors": [0]},
            TypeError,
            "received_contributors",
        ),
        (
            "ring",
            {"received_contributors": (0, 0)},
            ValueError,
            "received_contributors",
        ),
        (
            "ring",
            {"send_chunk": (0, 1)},
            TypeError,
            "send_chunk",
        ),
        (
            "tree",
            {"child_rank": False},
            TypeError,
            "child_rank",
        ),
        (
            "tree",
            {"parent_rank": -1},
            ValueError,
            "parent_rank",
        ),
        (
            "tree",
            {"parent_rank": 1},
            ValueError,
            "must differ",
        ),
    ),
)
def test_public_topology_metadata_rejects_invalid_local_values(
    constructor: str,
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import ChunkRange
    from ccdl_comm.cuda.transports.pipelined_ring import RingReduceScatterStep
    from ccdl_comm.cuda.transports.tree import TreeEdge

    if constructor == "ring":
        values: dict[str, object] = {
            "step_index": 0,
            "send_peer": 0,
            "recv_peer": 1,
            "send_chunk_owner": 0,
            "recv_chunk_owner": 1,
            "received_contributors": (0,),
            "send_chunk": ChunkRange(0, 1),
            "recv_chunk": ChunkRange(1, 2),
        }
        values.update(kwargs)
        with pytest.raises(exception, match=message):
            RingReduceScatterStep(**values)  # type: ignore[arg-type]
    else:
        values = {"child_rank": 1, "parent_rank": 0}
        values.update(kwargs)
        with pytest.raises(exception, match=message):
            TreeEdge(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        ("chunk_plan", {"world_size": 0}, "world_size"),
        ("ring", {"world_size": 3, "rank": -1}, "rank"),
        ("ring", {"world_size": 3, "rank": 3}, "rank"),
        ("tree", {"world_size": 3, "rank": 3, "root": 0}, "rank"),
        ("tree", {"world_size": 3, "rank": 0, "root": -1}, "root"),
        ("tree", {"world_size": 3, "rank": 0, "root": 3}, "root"),
    ),
)
def test_topology_schedule_factories_reject_invalid_values(factory: str, kwargs: dict[str, int], message: str) -> None:
    from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
    from ccdl_comm.cuda.transports.pipelined_ring import compile_pipelined_ring_schedule
    from ccdl_comm.cuda.transports.tree import compile_tree_schedule

    world_size = kwargs["world_size"]
    with pytest.raises(ValueError, match=message):
        if factory == "chunk_plan":
            compile_chunk_plan(original_numel=1, world_size=world_size)
        else:
            plan = compile_chunk_plan(original_numel=world_size, world_size=world_size)
            if factory == "ring":
                compile_pipelined_ring_schedule(chunk_plan=plan, rank=kwargs["rank"])
            else:
                compile_tree_schedule(chunk_plan=plan, rank=kwargs["rank"], root=kwargs["root"])
