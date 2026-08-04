from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ccdl_comm import CommunicationPlan, CommunicationStage, CompileContext, WorkspacePolicy


def test_plan_is_immutable_and_requires_explicit_strategy() -> None:
    plan = CommunicationPlan(collective="all_reduce", strategy="ring")

    with pytest.raises(FrozenInstanceError):
        plan.strategy = "auto"  # type: ignore[misc]


def test_hierarchical_plan_requires_stages() -> None:
    with pytest.raises(ValueError, match="requires at least one stage"):
        CommunicationPlan(collective="all_reduce", strategy="hierarchical")


def test_plan_copies_mutable_sequences_into_immutable_tuples() -> None:
    stages = [CommunicationStage("intra", "all_reduce", "ring")]
    fallback = ["all_gather"]

    plan = CommunicationPlan("all_reduce", "hierarchical", stages=stages, fallback=fallback)  # type: ignore[arg-type]
    stages.clear()
    fallback.clear()

    assert tuple(stage.name for stage in plan.stages) == ("intra",)
    assert plan.fallback == ("all_gather",)


def test_context_rejects_rank_outside_world() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        CompileContext(rank=4, world_size=4, device="cuda:0", shape=(1024,), dtype="float16")


def test_context_owns_immutable_process_group_mapping() -> None:
    groups = {"intra": object()}
    context = CompileContext(
        rank=0,
        world_size=2,
        device="cuda:0",
        shape=[1024],  # type: ignore[arg-type]
        dtype="float16",
        process_groups=groups,
    )
    groups.clear()

    assert context.shape == (1024,)
    assert tuple(context.process_groups) == ("intra",)
    with pytest.raises(TypeError):
        context.process_groups["inter"] = object()  # type: ignore[index]


def test_workspace_policy_rejects_invalid_cache_limits() -> None:
    with pytest.raises(ValueError, match="max_cached_bytes"):
        WorkspacePolicy(max_cached_bytes=-1)
    with pytest.raises(ValueError, match="max_entries"):
        WorkspacePolicy(max_entries=0)
