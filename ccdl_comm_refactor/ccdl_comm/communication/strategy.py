from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_STRATEGIES = {"auto", "all_gather", "all_reduce", "reduce_scatter", "hierarchical"}


@dataclass(frozen=True)
class TopologyInfo:
    """Distributed topology hints for strategy planning.

    All fields are optional except global rank and world size. Callers can pass
    partial topology information; the planner must fall back safely when it
    cannot prove that a fast path is supported.
    """

    rank: int
    world_size: int
    local_rank: int | None = None
    local_world_size: int | None = None
    node_id: int | None = None
    node_count: int | None = None
    intra_node: str = "unknown"
    inter_node: str = "unknown"
    has_process_groups: bool = False


@dataclass(frozen=True)
class CollectiveCapabilities:
    """Capability flags detected from transports, extensions, and runtime."""

    reduce_scatter: bool = False
    hierarchical: bool = False
    fused_dequant_reduce: bool = False
    capability_flags: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyPlan:
    """Result of resolving a requested compressed communication strategy."""

    requested_strategy: str
    strategy: str
    fallback_strategy: str
    reason: str
    requires_fallback: bool
    capability_flags: dict[str, bool]


def plan_ddp_compression_strategy(
    *,
    requested_strategy: str,
    world_size: int,
    rank: int = 0,
    local_world_size: int | None = None,
    node_count: int | None = None,
    bucket_numel: int = 0,
    topology: TopologyInfo | None = None,
    capabilities: CollectiveCapabilities | None = None,
) -> StrategyPlan:
    """Resolve a compressed communication strategy without torch dependency."""

    normalized = requested_strategy.strip().lower()
    if normalized not in SUPPORTED_STRATEGIES:
        return StrategyPlan(
            requested_strategy=requested_strategy,
            strategy="all_gather",
            fallback_strategy="all_gather",
            reason=f"unsupported strategy {requested_strategy!r}; falling back to all_gather",
            requires_fallback=True,
            capability_flags={},
        )

    active_capabilities = capabilities or CollectiveCapabilities()
    flags = _capability_flags(active_capabilities)
    if normalized == "hierarchical" and not active_capabilities.hierarchical:
        return _fallback("hierarchical transport unavailable for explicit strategy", flags, requested_strategy=normalized)
    if normalized == "reduce_scatter" and not active_capabilities.reduce_scatter:
        return _fallback("reduce_scatter transport unavailable for explicit strategy", flags, requested_strategy=normalized)
    if normalized != "auto":
        return StrategyPlan(
            requested_strategy=normalized,
            strategy=normalized,
            fallback_strategy="all_gather",
            reason=f"explicit strategy {normalized}",
            requires_fallback=False,
            capability_flags=flags,
        )

    active_topology = topology or TopologyInfo(
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
    )
    active_node_count = active_topology.node_count if active_topology.node_count is not None else node_count
    if world_size <= 2:
        return StrategyPlan(
            requested_strategy="auto",
            strategy="all_gather",
            fallback_strategy="all_gather",
            reason="world_size<=2 uses validated all_gather path",
            requires_fallback=False,
            capability_flags=flags,
        )

    if active_node_count is not None and active_node_count > 1:
        if not active_capabilities.hierarchical:
            return _fallback("multi-node hierarchical unavailable", flags)
        if not active_topology.has_process_groups:
            return _fallback("process groups unavailable for multi-node hierarchical strategy", flags)
        return StrategyPlan(
            requested_strategy="auto",
            strategy="hierarchical",
            fallback_strategy="all_gather",
            reason="multi-node hierarchical strategy selected",
            requires_fallback=False,
            capability_flags=flags,
        )

    if active_capabilities.reduce_scatter:
        return StrategyPlan(
            requested_strategy="auto",
            strategy="reduce_scatter",
            fallback_strategy="all_gather",
            reason="single-node capable reduce_scatter strategy selected",
            requires_fallback=False,
            capability_flags=flags,
        )
    if active_capabilities.hierarchical:
        return StrategyPlan(
            requested_strategy="auto",
            strategy="hierarchical",
            fallback_strategy="all_gather",
            reason="single-node hierarchical strategy selected",
            requires_fallback=False,
            capability_flags=flags,
        )

    return _fallback("reduce_scatter and hierarchical unavailable for single-node auto strategy", flags)


def _capability_flags(capabilities: CollectiveCapabilities) -> dict[str, bool]:
    flags = {
        "reduce_scatter": capabilities.reduce_scatter,
        "hierarchical": capabilities.hierarchical,
        "fused_dequant_reduce": capabilities.fused_dequant_reduce,
    }
    flags.update(capabilities.capability_flags)
    return flags


def _fallback(reason: str, flags: dict[str, bool], *, requested_strategy: str = "auto") -> StrategyPlan:
    return StrategyPlan(
        requested_strategy=requested_strategy,
        strategy="all_gather",
        fallback_strategy="all_gather",
        reason=f"{reason}; falling back to all_gather",
        requires_fallback=True,
        capability_flags=flags,
    )
