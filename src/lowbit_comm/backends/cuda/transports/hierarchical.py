"""Compile hierarchy membership while leaving process groups runtime-owned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lowbit_comm.compiler.passes.topology import Topology


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
