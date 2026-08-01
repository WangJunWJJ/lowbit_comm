"""Pure control-plane resolution for immutable communication plans."""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import UnsupportedCollective
from .plan import CommunicationPlan, CompileContext
from .registry import BackendKey, BackendRegistry


@dataclass(frozen=True)
class ResolvedPlan:
    """The strategy decision produced before backend compilation."""

    requested_strategy: str
    executed_strategy: str
    fallback_used: bool
    fallback_reason: str | None


def resolve_plan(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
) -> ResolvedPlan:
    """Resolve one plan without importing a tensor framework or GPU runtime."""

    if not isinstance(plan, CommunicationPlan):
        raise TypeError("plan must be a CommunicationPlan")
    if not isinstance(context, CompileContext):
        raise TypeError("context must be a CompileContext")
    if not isinstance(registry, BackendRegistry):
        raise TypeError("registry must be a BackendRegistry")

    requested_key = _key(plan, plan.strategy)
    if requested_key in registry:
        return ResolvedPlan(
            requested_strategy=plan.strategy,
            executed_strategy=plan.strategy,
            fallback_used=False,
            fallback_reason=None,
        )

    for fallback_strategy in plan.fallback:
        if _key(plan, fallback_strategy) in registry:
            return ResolvedPlan(
                requested_strategy=plan.strategy,
                executed_strategy=fallback_strategy,
                fallback_used=True,
                fallback_reason=f"explicit fallback from {plan.strategy} to {fallback_strategy}",
            )

    if plan.strategy != "auto":
        reason = "explicit strategy unavailable"
        if plan.fallback:
            reason = f"{reason} and no supported fallback was declared"
        else:
            reason = f"{reason} and no fallback was declared"
        raise UnsupportedCollective(
            f"{plan.collective}:{plan.strategy}",
            reason=reason,
        )

    return _resolve_auto(plan, context, registry)


def _key(plan: CommunicationPlan, strategy: str) -> BackendKey:
    return BackendKey(
        collective=plan.collective,
        strategy=strategy,
        backend=plan.backend,
        output_layout=plan.output_layout,
    )


def _resolve_auto(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
) -> ResolvedPlan:
    matching_strategies = {
        key.strategy
        for key in registry.keys()
        if key.collective == plan.collective
        and key.backend == plan.backend
        and key.output_layout == plan.output_layout
        and key.strategy != "auto"
    }
    if not matching_strategies:
        raise UnsupportedCollective(
            f"{plan.collective}:auto",
            reason="auto strategy found no registered backend for the requested dimensions",
        )

    preferred = _auto_priority(context)
    candidates = (*preferred, *sorted(matching_strategies.difference(preferred)))
    executed = next(strategy for strategy in candidates if strategy in matching_strategies)
    preferred_strategy = preferred[0]
    fallback_used = executed != preferred_strategy
    reason = None
    if fallback_used:
        reason = (
            f"auto fallback from unavailable preferred strategy "
            f"{preferred_strategy} to {executed}"
        )
    return ResolvedPlan(
        requested_strategy="auto",
        executed_strategy=executed,
        fallback_used=fallback_used,
        fallback_reason=reason,
    )


def _auto_priority(context: CompileContext) -> tuple[str, ...]:
    if context.world_size <= 2:
        return ("all_gather", "all_reduce", "topology", "ring", "reduce_scatter", "hierarchical")
    if context.node_count is not None and context.node_count > 1 and context.process_groups:
        return ("hierarchical", "reduce_scatter", "all_gather", "topology", "all_reduce", "ring")
    return ("reduce_scatter", "hierarchical", "all_gather", "topology", "all_reduce", "ring")
