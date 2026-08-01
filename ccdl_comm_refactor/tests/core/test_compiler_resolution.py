from __future__ import annotations

import pytest

from ccdl_comm import CommunicationPlan, CompileContext
from ccdl_comm.compiler import resolve_plan
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.registry import BackendKey, BackendRegistry


CONTEXT = CompileContext(
    rank=0,
    world_size=4,
    device="cuda:0",
    shape=(1024,),
    dtype="float16",
)


def _register(registry: BackendRegistry, strategy: str) -> None:
    registry.register(
        BackendKey("all_reduce", strategy, "cuda", "full"),
        lambda: object(),
    )


def test_explicit_registered_strategy_is_preserved() -> None:
    registry = BackendRegistry()
    _register(registry, "ring")

    resolved = resolve_plan(
        CommunicationPlan(collective="all_reduce", strategy="ring"),
        CONTEXT,
        registry,
    )

    assert resolved.requested_strategy == "ring"
    assert resolved.executed_strategy == "ring"
    assert resolved.fallback_used is False
    assert resolved.fallback_reason is None


def test_explicit_unsupported_strategy_does_not_silently_fallback() -> None:
    plan = CommunicationPlan(collective="all_reduce", strategy="ring")

    with pytest.raises(UnsupportedCollective, match="all_reduce:ring"):
        resolve_plan(plan, CONTEXT, BackendRegistry())


def test_explicit_fallback_uses_first_supported_entry() -> None:
    registry = BackendRegistry()
    _register(registry, "all_gather")
    plan = CommunicationPlan(
        collective="all_reduce",
        strategy="ring",
        fallback=("tree", "all_gather", "all_reduce"),
    )

    resolved = resolve_plan(plan, CONTEXT, registry)

    assert resolved.requested_strategy == "ring"
    assert resolved.executed_strategy == "all_gather"
    assert resolved.fallback_used is True
    assert resolved.fallback_reason == "explicit fallback from ring to all_gather"


def test_explicit_strategy_rejects_unsupported_declared_fallbacks() -> None:
    plan = CommunicationPlan(
        collective="all_reduce",
        strategy="ring",
        fallback=("tree", "all_gather"),
    )

    with pytest.raises(UnsupportedCollective, match="no supported fallback"):
        resolve_plan(plan, CONTEXT, BackendRegistry())


def test_auto_selects_registered_strategy_with_diagnostic_reason() -> None:
    registry = BackendRegistry()
    _register(registry, "all_gather")

    resolved = resolve_plan(
        CommunicationPlan(collective="all_reduce", strategy="auto"),
        CONTEXT,
        registry,
    )

    assert resolved.requested_strategy == "auto"
    assert resolved.executed_strategy == "all_gather"
    assert resolved.fallback_used is True
    assert resolved.fallback_reason is not None
    assert "auto" in resolved.fallback_reason


def test_auto_raises_when_no_matching_backend_is_registered() -> None:
    with pytest.raises(UnsupportedCollective, match="auto strategy found no registered backend"):
        resolve_plan(
            CommunicationPlan(collective="all_reduce", strategy="auto"),
            CONTEXT,
            BackendRegistry(),
        )
