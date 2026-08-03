"""Pure control-plane resolution for immutable communication plans."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock

from .backend import BackendCapabilities, StrategyChoice
from .exceptions import BackendRegistrationError, UnsupportedCollective
from .execution_info import ExecutionInfo
from .executor import (
    CompileCacheKey,
    CompiledCommunicationPlan,
    CompiledExecutor,
    ObjectIdentity,
)
from .plan import CommunicationPlan, CompileContext
from .registry import BackendKey, BackendRegistry


@dataclass(frozen=True)
class ResolvedPlan:
    """The strategy decision produced before backend compilation."""

    requested_strategy: str
    executed_strategy: str
    fallback_used: bool
    fallback_reason: str | None
    selection_reason: str | None = None
    strategy_policy_id: str | None = None
    benchmark_matched: bool = False
    strategy_evidence: str | None = None
    expected_speedup: float | None = None
    observed_speedup: float | None = None
    comparison_baseline: str | None = None


class CompileCache:
    """A bounded, thread-safe LRU cache for compiled communication plans."""

    def __init__(self, *, max_entries: int = 128) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._entries: OrderedDict[CompileCacheKey, CompiledCommunicationPlan] = OrderedDict()
        self._lock = RLock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def get(self, key: CompileCacheKey) -> CompiledCommunicationPlan | None:
        """Return and promote one entry, or ``None`` on a cache miss."""

        if not isinstance(key, CompileCacheKey):
            raise TypeError("key must be a CompileCacheKey")
        with self._lock:
            compiled = self._entries.get(key)
            if compiled is not None:
                self._entries.move_to_end(key)
            return compiled

    def put(self, key: CompileCacheKey, compiled: CompiledCommunicationPlan) -> None:
        """Insert one plan and evict the least-recently-used entry if needed."""

        if not isinstance(key, CompileCacheKey):
            raise TypeError("key must be a CompileCacheKey")
        if not isinstance(compiled, CompiledCommunicationPlan):
            raise TypeError("compiled must be a CompiledCommunicationPlan")
        with self._lock:
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get_or_create(
        self,
        key: CompileCacheKey,
        factory: Callable[[], CompiledCommunicationPlan],
    ) -> CompiledCommunicationPlan:
        """Atomically return an entry or create it exactly once for this cache."""

        if not isinstance(key, CompileCacheKey):
            raise TypeError("key must be a CompileCacheKey")
        if not callable(factory):
            raise TypeError("factory must be callable")
        with self._lock:
            compiled = self._entries.get(key)
            if compiled is not None:
                self._entries.move_to_end(key)
                return compiled
            compiled = factory()
            if not isinstance(compiled, CompiledCommunicationPlan):
                raise TypeError("factory must return a CompiledCommunicationPlan")
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return compiled

    def clear(self) -> None:
        """Release all cache-owned compiled plan references."""

        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_DEFAULT_REGISTRY = BackendRegistry()


def compile(
    plan: CommunicationPlan,
    context: CompileContext,
    *,
    registry: BackendRegistry | None = None,
    cache: CompileCache | None = None,
) -> CompiledCommunicationPlan:
    """Resolve and compile a plan once for repeated data-path execution."""

    active_registry = registry if registry is not None else _DEFAULT_REGISTRY
    if cache is not None and not isinstance(cache, CompileCache):
        raise TypeError("cache must be a CompileCache or None")

    resolved, effective_plan, backend = _select_backend(
        plan,
        context,
        active_registry,
    )
    cache_key = _compile_cache_key(
        plan,
        effective_plan,
        context,
        active_registry,
        resolved,
    )
    if cache is not None:
        return cache.get_or_create(
            cache_key,
            lambda: _compile_backend(
                backend,
                effective_plan,
                context,
                resolved,
                cache_key,
            ),
        )
    return _compile_backend(
        backend,
        effective_plan,
        context,
        resolved,
        cache_key,
    )


def _compile_backend(
    backend: object,
    effective_plan: CommunicationPlan,
    context: CompileContext,
    resolved: ResolvedPlan,
    cache_key: CompileCacheKey,
) -> CompiledCommunicationPlan:
    backend_key = _key(effective_plan, effective_plan.strategy)

    executor = backend.compile(effective_plan, context)
    if not isinstance(executor, CompiledExecutor):
        raise BackendRegistrationError(
            f"backend for {backend_key} did not return a CompiledExecutor"
        )
    if not isinstance(executor.execution_info, ExecutionInfo):
        raise BackendRegistrationError(
            f"executor for {backend_key} did not expose ExecutionInfo"
        )
    if executor.execution_info.backend != effective_plan.backend:
        raise BackendRegistrationError(
            f"executor for {backend_key} reported inconsistent backend "
            f"{executor.execution_info.backend!r}"
        )
    if executor.execution_info.executed_strategy != effective_plan.strategy:
        raise BackendRegistrationError(
            f"executor for {backend_key} reported inconsistent executed strategy "
            f"{executor.execution_info.executed_strategy!r}"
        )

    details = dict(executor.execution_info.details)
    if resolved.selection_reason is not None:
        details.update(
            {
                "strategy_selection_reason": resolved.selection_reason,
                "strategy_policy_id": resolved.strategy_policy_id,
                "strategy_benchmark_matched": resolved.benchmark_matched,
                "strategy_evidence": resolved.strategy_evidence,
                "strategy_expected_speedup": resolved.expected_speedup,
                "strategy_observed_speedup": resolved.observed_speedup,
                "strategy_comparison_baseline": resolved.comparison_baseline,
            }
        )
    execution_info = replace(
        executor.execution_info,
        requested_strategy=resolved.requested_strategy,
        executed_strategy=resolved.executed_strategy,
        fallback_used=resolved.fallback_used,
        fallback_reason=resolved.fallback_reason,
        details=details,
    )
    try:
        executor.execution_info = execution_info
    except (AttributeError, TypeError) as exc:
        raise BackendRegistrationError(
            f"executor for {backend_key} must allow compile-time ExecutionInfo binding"
        ) from exc
    if executor.execution_info is not execution_info:
        raise BackendRegistrationError(
            f"executor for {backend_key} did not retain bound ExecutionInfo"
        )
    compiled = CompiledCommunicationPlan(
        executor=executor,
        execution_info=execution_info,
        cache_key=cache_key,
    )
    return compiled


def _select_backend(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
) -> tuple[ResolvedPlan, CommunicationPlan, object]:
    rejected: list[str] = []
    choice = _auto_strategy_choice(plan, context, registry)
    for strategy, source in _compile_candidates(plan, context, registry, choice):
        backend_key = _key(plan, strategy)
        if backend_key not in registry:
            rejected.append(f"{strategy}: backend key is not registered")
            continue
        backend = registry.resolve(backend_key)
        if backend.name != plan.backend:
            raise BackendRegistrationError(
                f"backend name {backend.name!r} does not match registry key {backend_key}"
            )
        capabilities = backend.capabilities(context)
        if not isinstance(capabilities, BackendCapabilities):
            raise BackendRegistrationError(
                f"backend for {backend_key} did not return BackendCapabilities"
            )
        if capabilities.backend != plan.backend:
            raise BackendRegistrationError(
                f"capability backend {capabilities.backend!r} does not match registry key {backend_key}"
            )
        effective_plan = replace(plan, strategy=strategy, fallback=())
        rejection = _capability_rejection(capabilities, effective_plan, context)
        if rejection is not None:
            rejected.append(f"{strategy}: {rejection}")
            continue
        resolved = _resolved_candidate(plan, context, strategy, source, choice)
        if plan.strategy == "auto" and resolved.fallback_used and rejected:
            resolved = replace(
                resolved,
                fallback_reason=(
                    f"{resolved.fallback_reason}; rejected candidates: "
                    + "; ".join(rejected)
                ),
            )
        return resolved, effective_plan, backend

    reason = "no registered backend supports the requested compile context"
    if rejected:
        reason = f"{reason}; " + "; ".join(rejected)
    raise UnsupportedCollective(f"{plan.collective}:{plan.strategy}", reason=reason)


def _compile_candidates(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
    choice: StrategyChoice | None = None,
) -> tuple[tuple[str, str], ...]:
    if plan.strategy != "auto":
        return _unique_candidates(
            ((plan.strategy, "requested"),),
            tuple(
                (strategy, "explicit")
                for strategy in plan.fallback
                if strategy != "auto"
            ),
        )

    if choice is not None:
        return _unique_candidates(
            ((choice.strategy, "auto"),),
            tuple((strategy, "auto_fallback") for strategy in choice.fallback),
            tuple(
                (strategy, "explicit")
                for strategy in plan.fallback
                if strategy != "auto"
            ),
        )

    matching = {
        key.strategy
        for key in registry.keys()
        if key.collective == plan.collective
        and key.backend == plan.backend
        and key.output_layout == plan.output_layout
    }
    autonomous = (
        *(_auto_priority(context)),
        *sorted(matching.difference({"auto", *plan.fallback, *_auto_priority(context)})),
    )
    return _unique_candidates(
        tuple(
            (strategy, "explicit")
            for strategy in plan.fallback
            if strategy != "auto"
        ),
        tuple((strategy, "auto") for strategy in autonomous),
    )


def _unique_candidates(*groups: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for group in groups:
        for strategy, source in group:
            if strategy not in seen:
                seen.add(strategy)
                candidates.append((strategy, source))
    return tuple(candidates)


def _resolved_candidate(
    plan: CommunicationPlan,
    context: CompileContext,
    strategy: str,
    source: str,
    choice: StrategyChoice | None = None,
) -> ResolvedPlan:
    if source == "requested":
        return ResolvedPlan(plan.strategy, strategy, False, None)
    if source == "explicit" and plan.strategy != "auto":
        return ResolvedPlan(
            plan.strategy,
            strategy,
            True,
            f"explicit fallback from {plan.strategy} to {strategy}",
        )
    if choice is not None:
        fallback_used = strategy != choice.strategy
        fallback_reason = None
        if fallback_used:
            fallback_reason = (
                f"auto fallback from selected strategy {choice.strategy} to {strategy}"
            )
        return ResolvedPlan(
            "auto",
            strategy,
            fallback_used,
            fallback_reason,
            selection_reason=choice.reason,
            strategy_policy_id=choice.policy_id,
            benchmark_matched=choice.benchmark_matched,
            strategy_evidence=choice.evidence,
            expected_speedup=choice.expected_speedup,
            observed_speedup=choice.observed_speedup,
            comparison_baseline=choice.baseline,
        )
    preferred = _auto_priority(context)[0]
    fallback_used = strategy != preferred
    reason = None
    if fallback_used:
        reason = f"auto fallback from unavailable preferred strategy {preferred} to {strategy}"
    return ResolvedPlan("auto", strategy, fallback_used, reason)


def _capability_rejection(
    capabilities: BackendCapabilities,
    plan: CommunicationPlan,
    context: CompileContext,
) -> str | None:
    if not capabilities.available:
        return capabilities.reason or "backend is unavailable"
    if plan.collective not in capabilities.collectives:
        return f"collective {plan.collective!r} is unsupported"
    if plan.strategy not in capabilities.strategies:
        return f"strategy {plan.strategy!r} is unsupported"
    supported_dtypes = {_normalize_dtype(dtype) for dtype in capabilities.dtypes}
    if _normalize_dtype(context.dtype) not in supported_dtypes:
        return f"dtype {context.dtype!r} is unsupported"
    if plan.compression is not None and plan.compression.bit not in capabilities.bits:
        return f"bit {plan.compression.bit} is unsupported"
    if plan.output_layout not in capabilities.output_layouts:
        return f"output layout {plan.output_layout!r} is unsupported"
    if plan.async_op and not capabilities.supports_async:
        return "async execution is unsupported"
    if context.allow_dynamic_shape and not capabilities.supports_dynamic_shape:
        return "dynamic shape is unsupported"
    return None


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower().removeprefix("torch.")
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }
    return aliases.get(normalized, normalized)


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
    if plan.strategy != "auto" and requested_key in registry:
        return ResolvedPlan(
            requested_strategy=plan.strategy,
            executed_strategy=plan.strategy,
            fallback_used=False,
            fallback_reason=None,
        )

    for fallback_strategy in plan.fallback:
        if fallback_strategy == "auto":
            continue
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

    return _resolve_auto(
        plan,
        context,
        registry,
        _auto_strategy_choice(plan, context, registry),
    )


def _auto_strategy_choice(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
) -> StrategyChoice | None:
    if plan.strategy != "auto":
        return None
    selector = registry.strategy_selector(plan.backend)
    if selector is None:
        return None
    choice = selector(plan, context)
    if not isinstance(choice, StrategyChoice):
        raise BackendRegistrationError(
            f"strategy selector for backend {plan.backend!r} did not return StrategyChoice"
        )
    return choice


def _key(plan: CommunicationPlan, strategy: str) -> BackendKey:
    return BackendKey(
        collective=plan.collective,
        strategy=strategy,
        backend=plan.backend,
        output_layout=plan.output_layout,
    )


def _compile_cache_key(
    requested_plan: CommunicationPlan,
    effective_plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
    resolved: ResolvedPlan,
) -> CompileCacheKey:
    compression = effective_plan.compression
    return CompileCacheKey(
        registry_identity=ObjectIdentity(registry),
        backend=effective_plan.backend,
        collective=effective_plan.collective,
        requested_strategy=requested_plan.strategy,
        executed_strategy=effective_plan.strategy,
        fallback=requested_plan.fallback,
        output_layout=effective_plan.output_layout,
        async_op=effective_plan.async_op,
        stage_signature=tuple(_stage_signature(stage) for stage in effective_plan.stages),
        shape_class=_shape_class(context),
        dtype=context.dtype,
        layout=context.layout,
        rank=context.rank,
        world_size=context.world_size,
        device=context.device,
        local_rank=context.local_rank,
        local_world_size=context.local_world_size,
        node_id=context.node_id,
        node_count=context.node_count,
        process_group_identity=ObjectIdentity(context.process_group),
        process_group_identities=tuple(
            (name, ObjectIdentity(group))
            for name, group in sorted(context.process_groups.items())
        ),
        bit=compression.bit if compression is not None else None,
        group_size=compression.group_size if compression is not None else None,
        compression=compression,
        topology_signature=context.topology_signature,
        device_architecture=context.device_architecture,
        strategy_policy_id=resolved.strategy_policy_id,
        workspace_budget_bytes=context.workspace_budget_bytes,
        allow_dynamic_shape=context.allow_dynamic_shape,
        workspace_policy=effective_plan.workspace_policy,
    )


def _stage_signature(stage: object) -> tuple[object, ...]:
    return (
        getattr(stage, "name"),
        getattr(stage, "collective"),
        getattr(stage, "strategy"),
        getattr(stage, "backend"),
        getattr(stage, "compression"),
        ObjectIdentity(getattr(stage, "process_group")),
        getattr(stage, "output_layout"),
        getattr(stage, "async_op"),
    )


def _shape_class(context: CompileContext) -> tuple[int, ...]:
    # CompileContext has no dynamic-capacity bound yet. Exact shapes are the
    # only safe reusable class until a backend can prove a wider allocation.
    return context.shape


def _resolve_auto(
    plan: CommunicationPlan,
    context: CompileContext,
    registry: BackendRegistry,
    choice: StrategyChoice | None = None,
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

    if choice is not None:
        candidates = (choice.strategy, *choice.fallback, *plan.fallback)
        try:
            executed = next(
                strategy
                for strategy in candidates
                if strategy != "auto" and strategy in matching_strategies
            )
        except StopIteration as exc:
            raise UnsupportedCollective(
                f"{plan.collective}:auto",
                reason=(
                    f"selected strategy {choice.strategy!r} and its declared "
                    "fallbacks are not registered"
                ),
            ) from exc
        return _resolved_candidate(plan, context, executed, "auto", choice)

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
