from __future__ import annotations

import pytest

from lowbit_comm.backends.cuda.transports.hierarchical import (
    bind_grouped_transport,
    execute_grouped_full_tensor,
    GroupedTransportCapability,
    require_grouped_transport_capability,
    HierarchyBindings,
    compile_hierarchy,
    execute_hierarchical_mean,
)
from lowbit_comm.backends.cuda.transports.ring import compile_ring_schedule
from lowbit_comm.backends.cuda.transports.tree import compile_tree_schedule
from lowbit_comm.compiler.passes.topology import (
    compile_grouped_reduction,
    parse_topology_signature,
)
from lowbit_comm.core import GroupedReductionPlan, Topology


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


@pytest.mark.parametrize("world_size", (3, 5, 8, 16, 64))
def test_grouped_reduction_limits_kernel_fan_in_for_any_world_size(
    world_size: int,
) -> None:
    per_node = 8
    signature = "node_ids=" + ",".join(
        str(rank // per_node) for rank in range(world_size)
    )
    topology = parse_topology_signature(signature, world_size=world_size)

    plan = compile_grouped_reduction(topology, max_fan_in=8)

    assert plan.world_size == world_size
    assert plan.max_fan_in == 8
    assert all(1 <= len(group) <= 8 for level in plan.levels for group in level)
    assert sorted(rank for group in plan.levels[0] for rank in group) == list(
        range(world_size)
    )
    for previous, current in zip(plan.levels, plan.levels[1:]):
        assert sorted(rank for group in current for rank in group) == sorted(
            group[0] for group in previous
        )
    assert len(plan.levels[-1]) == 1
    assert plan.root == plan.levels[-1][0][0]


def test_grouped_reduction_preserves_uneven_node_locality() -> None:
    topology = parse_topology_signature(
        "node_ids=0,0,0,1,1,2,2,2,2",
        world_size=9,
    )

    plan = compile_grouped_reduction(topology, max_fan_in=4)

    assert plan.levels[0] == ((0, 1, 2), (3, 4), (5, 6, 7, 8))
    assert plan.levels[1] == ((0, 3, 5),)


def test_grouped_reduction_rejects_missing_or_repeated_rank_coverage() -> None:
    with pytest.raises(ValueError, match="cover every rank"):
        GroupedReductionPlan(
            world_size=4,
            max_fan_in=4,
            levels=(((0, 1, 1, 3),),),
        )


def test_grouped_transport_binds_process_groups_in_global_deterministic_order() -> None:
    topology = parse_topology_signature(
        "node_ids=0,0,0,1,1,2,2,2,2",
        world_size=9,
    )
    plan = compile_grouped_reduction(topology, max_fan_in=4)
    calls: list[tuple[int, ...]] = []

    bindings = bind_grouped_transport(
        plan,
        new_group=lambda ranks: calls.append(tuple(ranks)) or f"group:{tuple(ranks)}",
    )

    assert calls == [(0, 1, 2), (3, 4), (5, 6, 7, 8), (0, 3, 5)]
    assert bindings.group((3, 4)) == "group:(3, 4)"
    assert bindings.group((0, 3, 5)) == "group:(0, 3, 5)"


def test_grouped_transport_uses_local_binding_for_single_rank_group() -> None:
    topology = parse_topology_signature("node_ids=0,1,1", world_size=3)
    plan = compile_grouped_reduction(topology, max_fan_in=2)
    calls: list[tuple[int, ...]] = []

    bindings = bind_grouped_transport(
        plan,
        new_group=lambda ranks: calls.append(tuple(ranks)) or object(),
    )

    assert (0,) not in calls
    assert bindings.group((0,)) is None


def test_grouped_full_tensor_executes_forward_and_reverse_rank_schedule() -> None:
    topology = parse_topology_signature(
        "node_ids=0,0,0,1,1,2,2,2,2",
        world_size=9,
    )
    plan = compile_grouped_reduction(topology, max_fan_in=4)
    bindings = bind_grouped_transport(plan, new_group=lambda ranks: tuple(ranks))
    calls: list[tuple[str, tuple[int, ...], int]] = []

    execute_grouped_full_tensor(
        object(),
        plan,
        bindings,
        rank=3,
        reduce_stage=lambda value, ranks, group, dst: calls.append(
            ("reduce", ranks, dst)
        ),
        broadcast_stage=lambda value, ranks, group, src: calls.append(
            ("broadcast", ranks, src)
        ),
        finalize_root=lambda value, divisor: calls.append(
            ("normalize", (divisor,), 0)
        ),
    )

    assert calls == [
        ("reduce", (3, 4), 3),
        ("reduce", (0, 3, 5), 0),
        ("broadcast", (0, 3, 5), 0),
        ("broadcast", (3, 4), 3),
    ]


def test_grouped_full_tensor_stops_forward_reduction_for_non_leader() -> None:
    topology = parse_topology_signature("node_ids=0,0,1,1,1", world_size=5)
    plan = compile_grouped_reduction(topology, max_fan_in=4)
    bindings = bind_grouped_transport(plan, new_group=lambda ranks: tuple(ranks))
    calls: list[tuple[str, tuple[int, ...]]] = []

    execute_grouped_full_tensor(
        object(),
        plan,
        bindings,
        rank=4,
        reduce_stage=lambda value, ranks, group, dst: calls.append(
            ("reduce", ranks)
        ),
        broadcast_stage=lambda value, ranks, group, src: calls.append(
            ("broadcast", ranks)
        ),
        finalize_root=lambda value, divisor: calls.append(("normalize", (divisor,))),
    )

    assert calls == [
        ("reduce", (2, 3, 4)),
        ("broadcast", (2, 3, 4)),
    ]


def test_grouped_full_tensor_normalizes_global_sum_only_on_root() -> None:
    topology = parse_topology_signature("node_ids=0,0,1,1,1", world_size=5)
    plan = compile_grouped_reduction(topology, max_fan_in=4)
    bindings = bind_grouped_transport(plan, new_group=lambda ranks: tuple(ranks))
    normalizations: list[int] = []

    execute_grouped_full_tensor(
        object(),
        plan,
        bindings,
        rank=plan.root,
        reduce_stage=lambda value, ranks, group, dst: None,
        broadcast_stage=lambda value, ranks, group, src: None,
        finalize_root=lambda value, divisor: normalizations.append(divisor),
    )

    assert normalizations == [5]


def test_grouped_transport_capability_requires_quantized_sum_and_broadcast() -> None:
    capability = GroupedTransportCapability(
        bit=8,
        group_size=64,
        quant_type="linear",
        max_fan_in=8,
        supports_reduce_sum=True,
        supports_quantized_broadcast=True,
        async_completion=True,
    )

    assert require_grouped_transport_capability(
        capability,
        bit=8,
        group_size=64,
        quant_type="linear",
        max_fan_in=8,
    ) is capability


@pytest.mark.parametrize(
    "overrides",
    (
        {"supports_reduce_sum": False},
        {"supports_quantized_broadcast": False},
        {"async_completion": False},
        {"bit": 4},
        {"group_size": 32},
        {"max_fan_in": 4},
    ),
)
def test_grouped_transport_rejects_incomplete_or_mismatched_capability(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "bit": 8,
        "group_size": 64,
        "quant_type": "linear",
        "max_fan_in": 8,
        "supports_reduce_sum": True,
        "supports_quantized_broadcast": True,
        "async_completion": True,
    }
    values.update(overrides)
    capability = GroupedTransportCapability(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="grouped transport capability"):
        require_grouped_transport_capability(
            capability,
            bit=8,
            group_size=64,
            quant_type="linear",
            max_fan_in=8,
        )

    with pytest.raises(ValueError, match="prior representative"):
        GroupedReductionPlan(
            world_size=4,
            max_fan_in=2,
            levels=(((0, 1), (2, 3)), ((0, 3),)),
        )
