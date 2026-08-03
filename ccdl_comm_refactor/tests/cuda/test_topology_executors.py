from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

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


class _FakeWorkspaceSession:
    def __init__(self) -> None:
        self.buffers = {"workspace": object()}
        self.release_calls: list[object] = []
        self.abort_calls = 0

    def release(self, *, completion: object) -> None:
        self.release_calls.append(completion)

    def abort(self) -> None:
        self.abort_calls += 1


class _FakeRingRuntime:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.handles: list[object] = []
        self.completion = _FakeCompletion(self.calls)

    def wait_for_producer(self, tensor: object) -> None:
        self.calls.append(("producer_wait", tensor))

    def quant_pack(self, tensor: object, chunk: object, workspace: object) -> object:
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
    ) -> tuple[object, object]:
        handle = object()
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
    ) -> None:
        self.calls.append(("fused_reduce", chunk, contributors, workspace))

    def record_completion(self) -> _FakeCompletion:
        self.calls.append("record_completion")
        return self.completion

    def wait(self) -> None:
        raise AssertionError("run() must not call runtime.wait()")

    def synchronize(self) -> None:
        raise AssertionError("run() must not call runtime.synchronize()")


class _FakeTreeRuntime:
    def __init__(self, *, fail_on_quant: int | None = None) -> None:
        self.calls: list[object] = []
        self.handles: list[object] = []
        self.completion = _FakeCompletion(self.calls)
        self._quant_count = 0
        self._fail_on_quant = fail_on_quant

    def wait_for_producer(self, tensor: object) -> None:
        self.calls.append(("producer_wait", tensor))

    def quant_pack(self, tensor: object, edge: object, workspace: object) -> object:
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
    ) -> object:
        handle = object()
        self.handles.append(handle)
        self.calls.append(("send", peer, edge, workspace))
        return handle

    def receive(
        self,
        *,
        peer: int,
        edge: object,
        workspace: object,
    ) -> tuple[object, object]:
        payload = ("received", edge)
        handle = object()
        self.handles.append(handle)
        self.calls.append(("receive", peer, edge, workspace))
        return payload, handle

    def fused_reduce(
        self,
        tensor: object,
        received: object,
        edge: object,
        workspace: object,
    ) -> None:
        self.calls.append(("fused_reduce", edge, workspace))

    def apply_broadcast(
        self,
        tensor: object,
        received: object,
        edge: object,
        workspace: object,
    ) -> None:
        self.calls.append(("apply_broadcast", edge, workspace))

    def record_completion(self) -> _FakeCompletion:
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

    assert session.abort_calls == 1
    assert session.release_calls == []
    assert "cpu_wait" not in runtime.calls


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
    assert session.release_calls == [runtime.completion]
    assert session.abort_calls == 0
    assert "cpu_wait" not in runtime.calls


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

    assert session.abort_calls == 1
    assert session.release_calls == []
    assert "record_completion" not in runtime.calls
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
