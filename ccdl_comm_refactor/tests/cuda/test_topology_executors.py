from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest


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
