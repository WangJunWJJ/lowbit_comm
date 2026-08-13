from __future__ import annotations

import pytest

from lowbit_comm.backends.cuda.transports.hierarchical import (
    HierarchyBindings,
    compile_hierarchy,
    execute_hierarchical_mean,
)
from lowbit_comm.backends.cuda.transports.ring import compile_ring_schedule
from lowbit_comm.backends.cuda.transports.tree import compile_tree_schedule
from lowbit_comm.compiler.passes.topology import parse_topology_signature
from lowbit_comm.core import Topology


def test_topology_signature_supports_uneven_nodes() -> None:
    topology = parse_topology_signature(
        "node_ids=0,0,1,1,1,2",
        world_size=6,
    )

    assert topology.node_groups == ((0, 1), (2, 3, 4), (5,))
    assert topology.leaders == (0, 2, 5)
    assert topology.group_for_rank(3) == (2, 3, 4)
    assert isinstance(topology, Topology)


def test_topology_signature_rejects_rank_count_mismatch() -> None:
    with pytest.raises(ValueError, match="world_size"):
        parse_topology_signature("node_ids=0,0,1", world_size=4)


def test_ring_schedule_is_general_and_has_deterministic_collective_order() -> None:
    schedules = [compile_ring_schedule(world_size=3, rank=rank) for rank in range(3)]

    assert all(len(schedule.reduce_scatter) == 2 for schedule in schedules)
    assert all(len(schedule.all_gather) == 2 for schedule in schedules)
    assert schedules[0].reduce_scatter[0].send_peer == 1
    assert schedules[0].reduce_scatter[0].recv_peer == 2
    for step_index in range(2):
        sent = sorted(
            (step.send_peer, step.send_chunk)
            for schedule in schedules
            for step in (schedule.reduce_scatter[step_index],)
        )
        received = sorted(
            (schedule.rank, step.recv_chunk)
            for schedule in schedules
            for step in (schedule.reduce_scatter[step_index],)
        )
        assert sent == received


def test_tree_schedule_supports_non_power_of_two_world_size_and_rotated_root() -> None:
    schedules = [
        compile_tree_schedule(world_size=6, rank=rank, root=2)
        for rank in range(6)
    ]

    assert schedules[2].parent is None
    assert schedules[2].children == (3, 4)
    assert schedules[5].children == ()
    assert sum(len(schedule.children) for schedule in schedules) == 5


def test_hierarchy_compiles_local_and_leader_groups_without_fixed_cardinality() -> None:
    topology = parse_topology_signature("node_ids=0,0,1,1,1,2", world_size=6)

    plan = compile_hierarchy(topology, rank=4)

    assert plan.local_group == (2, 3, 4)
    assert plan.local_leader == 2
    assert plan.leader_group == (0, 2, 5)
    assert plan.is_leader is False
    assert plan.stage_order == ("intra_reduce", "inter_reduce", "intra_broadcast")


def test_hierarchy_runtime_obeys_compiled_collective_order() -> None:
    class Handle:
        def __init__(self, calls: list[str], name: str) -> None:
            self.calls = calls
            self.name = name

        def wait(self) -> None:
            self.calls.append(f"wait:{self.name}")

    class Dist:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reduce(self, tensor, *, dst, group, async_op):
            del tensor, dst, group, async_op
            self.calls.append("submit:intra_reduce")
            return Handle(self.calls, "intra_reduce")

        def all_reduce(self, tensor, *, group, async_op):
            del tensor, group, async_op
            self.calls.append("submit:inter_reduce")
            return Handle(self.calls, "inter_reduce")

        def broadcast(self, tensor, *, src, group, async_op):
            del tensor, src, group, async_op
            self.calls.append("submit:intra_broadcast")
            return Handle(self.calls, "intra_broadcast")

    class Tensor:
        def div_(self, divisor):
            assert divisor == 4
            return self

    topology = parse_topology_signature("node_ids=0,0,1,1", world_size=4)
    plan = compile_hierarchy(topology, rank=0)
    dist = Dist()

    execute_hierarchical_mean(
        Tensor(),
        plan,
        HierarchyBindings("local", "leaders"),
        world_size=4,
        dist=dist,
    )

    assert dist.calls == [
        "submit:intra_reduce",
        "wait:intra_reduce",
        "submit:inter_reduce",
        "wait:inter_reduce",
        "submit:intra_broadcast",
        "wait:intra_broadcast",
    ]
