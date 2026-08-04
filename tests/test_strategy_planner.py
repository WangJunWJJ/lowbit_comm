import pytest

from ccdl_comm.communication.strategy import (
    CollectiveCapabilities,
    TopologyInfo,
    plan_ddp_compression_strategy,
)
from ccdl_comm.exceptions import UnsupportedCollective


def test_legacy_auto_remains_conservative_without_compiled_bucket_context() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=2,
        rank=0,
        bucket_numel=33_554_432,
        capabilities=CollectiveCapabilities(reduce_scatter=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.fallback_strategy == "all_gather"
    assert plan.requires_fallback is False
    assert "world_size<=2" in plan.reason


def test_auto_falls_back_without_reduce_scatter_on_single_node_four_ranks() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(reduce_scatter=False),
    )

    assert plan.strategy == "all_gather"
    assert plan.requires_fallback is True
    assert "reduce_scatter and hierarchical unavailable" in plan.reason


def test_auto_selects_reduce_scatter_when_single_node_capable() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(reduce_scatter=True),
    )

    assert plan.strategy == "reduce_scatter"
    assert plan.requires_fallback is False
    assert "single-node capable" in plan.reason


def test_auto_falls_back_when_hierarchical_transport_is_not_performance_recommended() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(reduce_scatter=False, hierarchical=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.requires_fallback is True
    assert "hierarchical transport is not performance-recommended" in plan.reason


def test_auto_selects_hierarchical_when_single_node_transport_is_recommended() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(
            reduce_scatter=False,
            hierarchical=True,
            hierarchical_recommended=True,
        ),
    )

    assert plan.strategy == "hierarchical"
    assert plan.requires_fallback is False
    assert "single-node hierarchical" in plan.reason


def test_auto_multi_node_requires_hierarchical_capability_and_groups() -> None:
    topology = TopologyInfo(
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_id=0,
        node_count=2,
        has_process_groups=True,
    )

    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=8,
        rank=3,
        topology=topology,
        capabilities=CollectiveCapabilities(hierarchical=True),
    )

    assert plan.strategy == "hierarchical"
    assert plan.requires_fallback is False
    assert "multi-node hierarchical" in plan.reason


def test_auto_multi_node_falls_back_without_process_groups() -> None:
    topology = TopologyInfo(
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_id=0,
        node_count=2,
        has_process_groups=False,
    )

    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=8,
        rank=3,
        topology=topology,
        capabilities=CollectiveCapabilities(hierarchical=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.requires_fallback is True
    assert "process groups unavailable" in plan.reason


def test_explicit_strategy_is_preserved() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="all_gather",
        world_size=8,
        rank=0,
        capabilities=CollectiveCapabilities(reduce_scatter=True, hierarchical=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.requested_strategy == "all_gather"
    assert plan.requires_fallback is False


@pytest.mark.parametrize("strategy", ["hierarchical", "reduce_scatter", "topology"])
def test_explicit_unavailable_strategy_raises(strategy: str) -> None:
    with pytest.raises(UnsupportedCollective, match=strategy):
        plan_ddp_compression_strategy(
            requested_strategy=strategy,
            world_size=4,
            capabilities=CollectiveCapabilities(),
        )


def test_unknown_explicit_strategy_raises() -> None:
    with pytest.raises(UnsupportedCollective, match="unknown"):
        plan_ddp_compression_strategy(
            requested_strategy="unknown",
            world_size=4,
        )
