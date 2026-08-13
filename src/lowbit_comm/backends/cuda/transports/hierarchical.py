"""Compile hierarchy membership while leaving process groups runtime-owned."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any

from lowbit_comm.core.topology import GroupedReductionPlan, Topology


@dataclass(frozen=True, slots=True)
class HierarchyPlan:
    rank: int
    local_group: tuple[int, ...]
    local_leader: int
    leader_group: tuple[int, ...]
    is_leader: bool
    stage_order: tuple[str, ...] = (
        "intra_reduce",
        "inter_reduce",
        "intra_broadcast",
    )


@dataclass(frozen=True, slots=True)
class HierarchyBindings:
    """Runtime-owned groups bound after the pure plan is compiled."""

    local_group: object
    leader_group: object


@dataclass(frozen=True, slots=True)
class GroupedTransportBindings:
    groups: tuple[tuple[tuple[int, ...], object | None], ...]

    def group(self, ranks: tuple[int, ...]) -> object | None:
        key = tuple(ranks)
        try:
            return dict(self.groups)[key]
        except KeyError as error:
            raise KeyError(f"unbound grouped transport ranks: {key}") from error


def bind_grouped_transport(
    plan: GroupedReductionPlan,
    *,
    new_group: Callable[[Sequence[int]], object],
) -> GroupedTransportBindings:
    if not isinstance(plan, GroupedReductionPlan):
        raise TypeError("plan must be a GroupedReductionPlan")
    ordered = tuple(group for level in plan.levels for group in level)
    if len(ordered) != len(set(ordered)):
        raise ValueError("grouped transport plan contains duplicate process groups")
    groups = tuple(
        (group, None if len(group) == 1 else new_group(group))
        for group in ordered
    )
    return GroupedTransportBindings(groups)


def execute_grouped_full_tensor(
    value: Any,
    plan: GroupedReductionPlan,
    bindings: GroupedTransportBindings,
    *,
    rank: int,
    reduce_stage: Callable[[Any, tuple[int, ...], object | None, int], None],
    broadcast_stage: Callable[[Any, tuple[int, ...], object | None, int], None],
    finalize_root: Callable[[Any, int], None],
) -> Any:
    if rank < 0 or rank >= plan.world_size:
        raise ValueError("rank must be within grouped reduction world_size")
    participated: list[tuple[int, ...]] = []
    active = True
    for level in plan.levels:
        group = next((group for group in level if rank in group), None)
        if group is None:
            continue
        if active:
            reduce_stage(value, group, bindings.group(group), group[0])
            participated.append(group)
            active = rank == group[0]
    if rank == plan.root:
        finalize_root(value, plan.world_size)
    for group in reversed(participated):
        broadcast_stage(value, group, bindings.group(group), group[0])
    return value


def execute_hierarchical_mean(
    tensor: Any,
    plan: HierarchyPlan,
    bindings: HierarchyBindings,
    *,
    world_size: int,
    dist: Any,
) -> Any:
    """Execute local reduce, leader reduce, and local broadcast in order."""

    local = dist.reduce(
        tensor,
        dst=plan.local_leader,
        group=bindings.local_group,
        async_op=True,
    )
    local.wait()
    if plan.is_leader:
        inter = dist.all_reduce(
            tensor,
            group=bindings.leader_group,
            async_op=True,
        )
        inter.wait()
        tensor.div_(world_size)
    broadcast = dist.broadcast(
        tensor,
        src=plan.local_leader,
        group=bindings.local_group,
        async_op=True,
    )
    broadcast.wait()
    return tensor




def compile_hierarchy(topology: Topology, *, rank: int) -> HierarchyPlan:
    if not isinstance(topology, Topology):
        raise TypeError("topology must be a Topology")
    local_group = topology.group_for_rank(rank)
    leader = local_group[0]
    return HierarchyPlan(
        rank=rank,
        local_group=local_group,
        local_leader=leader,
        leader_group=topology.leaders,
        is_leader=rank == leader,
    )
