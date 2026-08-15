"""Evidence-gated compiler orchestration and conservative fallback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, NamedTuple, cast

from lowbit_comm.api.intent import (
    CommunicationIntent,
    OutputSemantics,
    ShapeFamily,
    TensorSpec,
    _validate_communication_intent_graph,
)
from lowbit_comm.api.policy import (
    AutoConstraints,
    AutoPolicy,
    CollectiveKind,
    ExplicitPolicy,
    NativePolicy,
    StrategySpec,
    _auto_constraints_allow,
    _canonical_native_strategy,
    _validate_auto_constraints_graph,
    _validate_policy_graph,
    _validate_strategy_graph,
)
from lowbit_comm.backends.protocols import BackendCapability, BackendPlan
from lowbit_comm.compiler.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceKey,
    EvidenceMetrics,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
    LegacyEvidenceMetrics,
    LegacyEvidenceRecord,
    _validate_evidence_record,
    _validate_evidence_store_records,
    _validate_legacy_evidence_record,
)
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.environment import EnvironmentFingerprint
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
    LowbitCommError,
)
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
    _resolve_static_callable_member,
    _same_bound_callable,
    _validate_compilation_context_graph,
)
from lowbit_comm.core.signatures import (
    _require_dataclass_field_coverage,
    strategy_key,
)


Policy = NativePolicy | AutoPolicy | ExplicitPolicy
CacheKey = tuple[str, str, str, str, str]
_CANONICAL_DATACLASS_FIELDS = {
    CommunicationIntent: frozenset(
        {
            "tensor",
            "shape_family",
            "reduction",
            "output",
            "completion",
            "world_size",
            "rank",
        }
    ),
    TensorSpec: frozenset({"dtype", "shape"}),
    ShapeFamily: frozenset({"max_numel", "alignment"}),
    StrategySpec: frozenset(
        {
            "compression",
            "collective",
            "topology",
            "group_size",
            "accumulation_dtype",
            "error_feedback",
            "parameter_error_feedback",
            "overlap",
            "workspace_budget_bytes",
        }
    ),
    AutoConstraints: frozenset(
        {
            "allowed_compressions",
            "denied_compressions",
            "allowed_collectives",
            "denied_collectives",
            "allowed_topologies",
            "denied_topologies",
            "max_workspace_bytes",
        }
    ),
    NativePolicy: frozenset(),
    AutoPolicy: frozenset({"constraints"}),
    ExplicitPolicy: frozenset({"strategy"}),
    EnvironmentFingerprint: frozenset({"dimensions"}),
    CompilationContext: frozenset(
        {
            "environment",
            "workspace_budget_bytes",
            "node_count",
            "workload_class",
            "bucket_min_bytes",
            "bucket_max_bytes",
        }
    ),
    EvidenceKey: frozenset({"schema_version", "dimensions"}),
    LegacyEvidenceMetrics: frozenset(
        {
            "communication_gain_percent",
            "end_to_end_gain_percent",
            "quality_loss_percent",
            "convergence_step_increase_percent",
            "worst_run_gain_percent",
            "seeds",
            "cross_workload_reproduced",
        }
    ),
    EvidenceMetrics: frozenset(
        {
            "communication_gain_percent",
            "exposed_communication_gain_percent",
            "end_to_end_gain_percent",
            "quality_loss_percent",
            "convergence_step_increase_percent",
            "worst_run_gain_percent",
            "seeds",
            "cross_workload_reproduced",
        }
    ),
    LegacyEvidenceRecord: frozenset(
        {"key", "strategy", "status", "metrics"}
    ),
    EvidenceRecord: frozenset(
        {"key", "strategy", "status", "metrics"}
    ),
}


@dataclass(frozen=True, slots=True)
class _BoundBackendPlan:
    """Public adapter around one compiler-bound backend execute callable."""

    _execute: Callable[[object], object]

    def __post_init__(self) -> None:
        if not callable(self._execute):
            raise CompileError("Bound backend execute must be callable.")

    def execute(self, value: object) -> object:
        return self._execute(value)


class _CachedPlanEntry(NamedTuple):
    """One trusted execution plan retained only inside Compiler cache."""

    cache_key: CacheKey
    intent: CommunicationIntent
    strategy: StrategySpec
    backend_id: str
    backend_plan: BackendPlan
    backend_plan_identity: int
    execute: Callable[[object], object]
    origin: PlanOrigin
    signature: str
    evidence_fingerprint: str | None


class Compiler:
    """Compile immutable intents using strict policy priority semantics."""

    def __init__(
        self,
        registry: BackendRegistry,
        evidence: EvidenceStore,
    ) -> None:
        if type(registry) is not BackendRegistry:
            raise CompileError("Compiler registry must be BackendRegistry.")
        if type(evidence) is not EvidenceStore:
            raise CompileError("Compiler evidence must be EvidenceStore.")
        self._registry = registry
        self._evidence = evidence
        self._cache: dict[CacheKey, _CachedPlanEntry] = {}

    def compile(
        self,
        intent: CommunicationIntent,
        policy: Policy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        """Resolve, validate, lower, and cache one exact compile request."""
        _validate_compile_inputs(intent, policy, context)
        trusted_intent = _snapshot_intent(intent)
        trusted_policy = _snapshot_policy(policy)
        trusted_context = _snapshot_context(context)
        evidence_generation = _evidence_generation(self._evidence)
        cache_key = (
            _fingerprint(_intent_data(trusted_intent)),
            _fingerprint(_policy_data(trusted_policy)),
            _fingerprint(_context_data(trusted_context)),
            str(self._registry.generation),
            evidence_generation,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            entry = _validate_cached_entry(
                cached,
                cache_key,
                trusted_intent,
                trusted_policy,
                trusted_context,
            )
            return _project_execution_plan(entry)

        strategy, origin, evidence_record = self._resolve(
            trusted_intent,
            trusted_policy,
            trusted_context,
        )
        trusted_strategy = _snapshot_strategy(strategy)
        _validate_strategy_context(
            trusted_intent,
            trusted_strategy,
            trusted_context,
        )
        capability, lower = _resolve_backend(
            self._registry,
            trusted_intent,
            trusted_strategy,
        )
        backend_plan = _lower_backend(
            lower,
            _snapshot_intent(trusted_intent),
            _snapshot_strategy(trusted_strategy),
        )
        execute = _resolve_static_callable_member(
            backend_plan,
            "execute",
            "Cached backend plan must provide callable execute().",
        )
        evidence_fingerprint = (
            None
            if evidence_record is None
            else _record_fingerprint(evidence_record)
        )
        entry_intent = _snapshot_intent(trusted_intent)
        entry_strategy = _snapshot_strategy(trusted_strategy)
        signature = _plan_signature(
            intent=entry_intent,
            strategy=entry_strategy,
            context=trusted_context,
            backend_id=capability.backend_id,
            origin=origin,
            evidence_fingerprint=evidence_fingerprint,
        )
        entry = _CachedPlanEntry(
            cache_key=cache_key,
            intent=entry_intent,
            strategy=entry_strategy,
            backend_id=capability.backend_id,
            backend_plan=backend_plan,
            backend_plan_identity=id(backend_plan),
            execute=execute,
            origin=origin,
            signature=signature,
            evidence_fingerprint=evidence_fingerprint,
        )
        _validate_cached_entry_structure(entry)
        self._cache[cache_key] = entry
        return _project_execution_plan(entry)

    def _resolve(
        self,
        intent: CommunicationIntent,
        policy: Policy,
        context: CompilationContext,
    ) -> tuple[StrategySpec, PlanOrigin, EvidenceRecord | None]:
        if type(policy) is NativePolicy:
            return _canonical_native_strategy(), PlanOrigin.NATIVE, None
        if type(policy) is ExplicitPolicy:
            return policy.strategy, PlanOrigin.EXPLICIT, None

        selected = _select_evidence_strategy(
            self._evidence,
            self._registry,
            intent,
            policy.constraints,
            context,
        )
        if selected is not None:
            strategy, record = selected
            return strategy, PlanOrigin.AUTO, record
        return (
            _canonical_native_strategy(),
            PlanOrigin.NATIVE_FALLBACK,
            None,
        )


def _snapshot_intent(intent: object) -> CommunicationIntent:
    """Return a complete independent snapshot of one caller intent."""
    source = _validate_communication_intent_graph(intent)
    _guard_canonical(source, CommunicationIntent)
    _guard_canonical(source.tensor, TensorSpec)
    _guard_canonical(source.shape_family, ShapeFamily)
    tensor = TensorSpec(
        dtype=source.tensor.dtype,
        shape=tuple(dimension for dimension in source.tensor.shape),
    )
    shape_family = ShapeFamily(
        max_numel=source.shape_family.max_numel,
        alignment=source.shape_family.alignment,
    )
    return CommunicationIntent(
        tensor=tensor,
        shape_family=shape_family,
        reduction=source.reduction,
        output=source.output,
        completion=source.completion,
        world_size=source.world_size,
        rank=source.rank,
    )


def _snapshot_strategy(strategy: object) -> StrategySpec:
    """Return a complete independent snapshot of one selected strategy."""
    source = _validate_strategy_graph(strategy)
    _guard_canonical(source, StrategySpec)
    return StrategySpec(
        compression=source.compression,
        collective=source.collective,
        topology=source.topology,
        group_size=source.group_size,
        accumulation_dtype=source.accumulation_dtype,
        error_feedback=source.error_feedback,
        parameter_error_feedback=source.parameter_error_feedback,
        overlap=source.overlap,
        workspace_budget_bytes=source.workspace_budget_bytes,
    )


def _snapshot_constraints(constraints: object) -> AutoConstraints:
    """Return an independent snapshot of every Auto constraint set."""
    source = _validate_auto_constraints_graph(constraints)
    _guard_canonical(source, AutoConstraints)
    return AutoConstraints(
        allowed_compressions=_snapshot_optional_frozenset(
            source.allowed_compressions
        ),
        denied_compressions=_snapshot_frozenset(
            source.denied_compressions
        ),
        allowed_collectives=_snapshot_optional_frozenset(
            source.allowed_collectives
        ),
        denied_collectives=_snapshot_frozenset(
            source.denied_collectives
        ),
        allowed_topologies=_snapshot_optional_frozenset(
            source.allowed_topologies
        ),
        denied_topologies=_snapshot_frozenset(
            source.denied_topologies
        ),
        max_workspace_bytes=source.max_workspace_bytes,
    )


def _snapshot_frozenset(values: frozenset[Any]) -> frozenset[Any]:
    return frozenset(value for value in values)


def _snapshot_optional_frozenset(
    values: frozenset[Any] | None,
) -> frozenset[Any] | None:
    if values is None:
        return None
    return _snapshot_frozenset(values)


def _snapshot_policy(policy: object) -> Policy:
    """Return a complete independent snapshot of one compile policy."""
    source = _validate_policy_graph(policy)
    _guard_canonical(source, type(source))
    if type(source) is NativePolicy:
        return NativePolicy()
    if type(source) is ExplicitPolicy:
        return ExplicitPolicy(_snapshot_strategy(source.strategy))
    return AutoPolicy(_snapshot_constraints(source.constraints))


def _snapshot_context(context: object) -> CompilationContext:
    """Return an independent snapshot of compilation environment data."""
    source = _validate_compilation_context_graph(context)
    _guard_canonical(source, CompilationContext)
    _guard_canonical(source.environment, EnvironmentFingerprint)
    environment = EnvironmentFingerprint(
        tuple(
            (key, value)
            for key, value in source.environment.dimensions
        )
    )
    return CompilationContext(
        environment=environment,
        workspace_budget_bytes=source.workspace_budget_bytes,
        node_count=source.node_count,
        workload_class=source.workload_class,
        bucket_min_bytes=source.bucket_min_bytes,
        bucket_max_bytes=source.bucket_max_bytes,
    )


def _validate_cached_entry(
    cached: object,
    cache_key: CacheKey,
    intent: CommunicationIntent,
    policy: Policy,
    context: CompilationContext,
) -> _CachedPlanEntry:
    """Freshly validate a private cache entry before public projection."""
    entry = _validate_cached_entry_structure(cached)
    if entry.cache_key != cache_key or entry.intent != intent:
        raise CompileError("Compiler cached execution plan is inconsistent.")
    _validate_cached_policy(entry, policy)
    _validate_strategy_context(entry.intent, entry.strategy, context)
    expected_signature = _plan_signature(
        intent=entry.intent,
        strategy=entry.strategy,
        context=context,
        backend_id=entry.backend_id,
        origin=entry.origin,
        evidence_fingerprint=entry.evidence_fingerprint,
    )
    if entry.signature != expected_signature:
        raise CompileError("Compiler cached plan signature is inconsistent.")
    return entry


def _validate_cached_entry_structure(cached: object) -> _CachedPlanEntry:
    """Validate one exact tuple-backed cache entry and execution anchor."""
    message = "Compiler cached execution plan is invalid."
    if type(cached) is not _CachedPlanEntry or len(cached) != 10:
        raise CompileError(message)
    entry = cast(_CachedPlanEntry, cached)
    if (
        type(entry.cache_key) is not tuple
        or len(entry.cache_key) != 5
        or not all(type(component) is str for component in entry.cache_key)
    ):
        raise CompileError("Cached plan key is invalid.")
    _validate_communication_intent_graph(entry.intent)
    _validate_strategy_graph(entry.strategy)
    if type(entry.backend_id) is not str or not entry.backend_id:
        raise CompileError("Cached plan backend identifier is invalid.")
    if (
        type(entry.backend_plan_identity) is not int
        or id(entry.backend_plan) != entry.backend_plan_identity
    ):
        raise CompileError("Cached backend plan identity is invalid.")
    resolved_execute = _resolve_static_callable_member(
        entry.backend_plan,
        "execute",
        "Cached backend plan must provide callable execute().",
    )
    if not _same_bound_callable(entry.execute, resolved_execute):
        raise CompileError("Cached backend execute binding is invalid.")
    if type(entry.origin) is not PlanOrigin:
        raise CompileError("Cached plan origin is invalid.")
    if type(entry.signature) is not str or not entry.signature:
        raise CompileError("Cached plan signature is invalid.")
    if entry.evidence_fingerprint is not None and type(
        entry.evidence_fingerprint
    ) is not str:
        raise CompileError("Cached evidence fingerprint is invalid.")
    return entry


def _validate_cached_policy(
    entry: _CachedPlanEntry,
    policy: Policy,
) -> None:
    """Require cached provenance to match the current policy value."""
    if type(policy) is NativePolicy:
        valid = (
            entry.origin is PlanOrigin.NATIVE
            and entry.strategy == _canonical_native_strategy()
            and entry.evidence_fingerprint is None
        )
    elif type(policy) is ExplicitPolicy:
        valid = (
            entry.origin is PlanOrigin.EXPLICIT
            and entry.strategy == policy.strategy
            and entry.evidence_fingerprint is None
        )
    elif entry.origin is PlanOrigin.AUTO:
        valid = (
            type(entry.evidence_fingerprint) is str
            and bool(entry.evidence_fingerprint)
            and _auto_constraints_allow(policy.constraints, entry.strategy)
        )
    else:
        valid = (
            entry.origin is PlanOrigin.NATIVE_FALLBACK
            and entry.strategy == _canonical_native_strategy()
            and entry.evidence_fingerprint is None
        )
    if not valid:
        raise CompileError("Compiler cached policy provenance is invalid.")


def _project_execution_plan(entry: _CachedPlanEntry) -> ExecutionPlan:
    """Return a fresh public wrapper without exposing cached plan state."""
    return ExecutionPlan(
        intent=_snapshot_intent(entry.intent),
        strategy=_snapshot_strategy(entry.strategy),
        backend_id=entry.backend_id,
        backend_plan=_BoundBackendPlan(entry.execute),
        origin=entry.origin,
        signature=entry.signature,
        evidence_fingerprint=entry.evidence_fingerprint,
    )


def _resolve_backend(
    registry: BackendRegistry,
    intent: CommunicationIntent,
    strategy: StrategySpec,
) -> tuple[
    BackendCapability,
    Callable[[CommunicationIntent, StrategySpec], BackendPlan],
]:
    candidates = registry.candidates(intent, strategy)
    if not candidates:
        raise CapabilityError(
            "No backend supports the exact intent and strategy."
        )
    return registry._resolve_lowering(candidates[0][0])


def _lower_backend(
    lower: Callable[[CommunicationIntent, StrategySpec], BackendPlan],
    intent: CommunicationIntent,
    strategy: StrategySpec,
) -> BackendPlan:
    """Invoke only the callable validated and bound by Registry."""
    try:
        return lower(intent, strategy)
    except LowbitCommError:
        raise
    except Exception as error:
        raise CompileError(
            "Backend lower() failed during compilation."
        ) from error


def _select_evidence_strategy(
    evidence: EvidenceStore,
    registry: BackendRegistry,
    intent: CommunicationIntent,
    constraints: AutoConstraints,
    context: CompilationContext,
) -> tuple[StrategySpec, EvidenceRecord] | None:
    records = sorted(
        _valid_evidence_records(evidence),
        key=lambda record: (
            record.key.schema_version,
            record.key.dimensions,
        ),
    )
    for record in records:
        if record.status is not EvidenceStatus.PRODUCTION_AUTO:
            continue
        try:
            strategy = record.strategy
            requested_key = EvidenceKey.from_request(
                environment=context.environment,
                intent=intent,
                strategy=strategy,
                node_count=context.node_count,
                workload_class=context.workload_class,
                bucket_min_bytes=context.bucket_min_bytes,
                bucket_max_bytes=context.bucket_max_bytes,
            )
        except CompileError:
            continue
        if evidence.production_auto_match(requested_key) is not record:
            continue
        if not _auto_constraints_allow(constraints, strategy):
            continue
        if not _strategy_context_is_valid(intent, strategy, context):
            continue
        if not registry.candidates(intent, strategy):
            continue
        return strategy, record
    return None


def _valid_evidence_records(
    evidence: EvidenceStore,
) -> tuple[EvidenceRecord, ...]:
    """Discard evidence that fails fresh trust-boundary validation."""
    valid_records: list[EvidenceRecord] = []
    for record in _validate_evidence_store_records(evidence):
        try:
            _validate_evidence_record(record)
        except CompileError:
            continue
        if record.key.schema_version == EVIDENCE_SCHEMA_VERSION:
            valid_records.append(record)
    return tuple(valid_records)


def _validate_compile_inputs(
    intent: CommunicationIntent,
    policy: Policy,
    context: CompilationContext,
) -> None:
    _validate_communication_intent_graph(intent)
    _validate_policy_graph(policy)
    _validate_compilation_context_graph(context)


def _strategy_context_is_valid(
    intent: CommunicationIntent,
    strategy: StrategySpec,
    context: CompilationContext,
) -> bool:
    try:
        _validate_strategy_context(intent, strategy, context)
    except CompileError:
        return False
    return True


def _validate_strategy_context(
    intent: CommunicationIntent,
    strategy: StrategySpec,
    context: CompilationContext,
) -> None:
    if (
        strategy.collective
        is CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
        and intent.output is not OutputSemantics.FULL_TENSOR
    ):
        raise CompileError(
            "Compressed all-gather reduce requires full-tensor output."
        )
    if strategy.parameter_error_feedback:
        raise CompileError(
            "Parameter error feedback requires an explicit sharded "
            "parameter protocol."
        )
    required_workspace = strategy.workspace_budget_bytes
    if (
        required_workspace is not None
        and required_workspace > context.workspace_budget_bytes
    ):
        raise CompileError(
            "Strategy workspace exceeds the compilation budget."
        )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _guard_canonical(
    value: object,
    expected_type: type[object],
) -> None:
    """Fail before serialization when one contract's field set drifts."""
    try:
        classified_fields = _CANONICAL_DATACLASS_FIELDS[expected_type]
    except (KeyError, TypeError) as error:
        raise CompileError(
            f"{expected_type.__name__} canonical fields require an update."
        ) from error
    _require_dataclass_field_coverage(
        value,
        expected_type,
        classified_fields,
        f"{expected_type.__name__} canonical fields require an update.",
    )


def _intent_data(intent: CommunicationIntent) -> dict[str, Any]:
    _guard_canonical(intent, CommunicationIntent)
    _guard_canonical(intent.tensor, TensorSpec)
    _guard_canonical(intent.shape_family, ShapeFamily)
    return {
        "completion": intent.completion.value,
        "output": intent.output.value,
        "rank": intent.rank,
        "reduction": intent.reduction.value,
        "shape_family": {
            "alignment": intent.shape_family.alignment,
            "max_numel": intent.shape_family.max_numel,
        },
        "tensor": {
            "dtype": intent.tensor.dtype,
            "shape": list(intent.tensor.shape),
        },
        "world_size": intent.world_size,
    }


def _canonical_strategy_data(strategy: StrategySpec) -> dict[str, Any]:
    _guard_canonical(strategy, StrategySpec)
    return {
        "canonical": [list(component) for component in strategy_key(strategy)]
    }


def _legacy_evidence_strategy_data(
    strategy: StrategySpec,
) -> dict[str, Any]:
    """Return the schema-v1 strategy encoding used by evidence hashes."""
    _guard_canonical(strategy, StrategySpec)
    return {
        "accumulation_dtype": strategy.accumulation_dtype.value,
        "collective": strategy.collective.value,
        "compression": strategy.compression.value,
        "error_feedback": strategy.error_feedback,
        "group_size": strategy.group_size,
        "overlap": strategy.overlap,
        "parameter_error_feedback": strategy.parameter_error_feedback,
        "topology": strategy.topology.value,
        "workspace_budget_bytes": strategy.workspace_budget_bytes,
    }


def _constraints_data(constraints: AutoConstraints) -> dict[str, Any]:
    _guard_canonical(constraints, AutoConstraints)

    def values(items: frozenset[Any] | None) -> list[str] | None:
        if items is None:
            return None
        return sorted(item.value for item in items)

    return {
        "allowed_collectives": values(constraints.allowed_collectives),
        "allowed_compressions": values(constraints.allowed_compressions),
        "allowed_topologies": values(constraints.allowed_topologies),
        "denied_collectives": values(constraints.denied_collectives),
        "denied_compressions": values(constraints.denied_compressions),
        "denied_topologies": values(constraints.denied_topologies),
        "max_workspace_bytes": constraints.max_workspace_bytes,
    }


def _policy_data(policy: Policy) -> dict[str, Any]:
    if type(policy) is NativePolicy:
        _guard_canonical(policy, NativePolicy)
        return {"kind": "native"}
    if type(policy) is ExplicitPolicy:
        _guard_canonical(policy, ExplicitPolicy)
        return {
            "kind": "explicit",
            "strategy": _canonical_strategy_data(policy.strategy),
        }
    _guard_canonical(policy, AutoPolicy)
    return {
        "constraints": _constraints_data(policy.constraints),
        "kind": "auto",
    }


def _context_data(context: CompilationContext) -> dict[str, Any]:
    _guard_canonical(context, CompilationContext)
    _guard_canonical(context.environment, EnvironmentFingerprint)
    return {
        "bucket_max_bytes": context.bucket_max_bytes,
        "bucket_min_bytes": context.bucket_min_bytes,
        "environment": [list(item) for item in context.environment.dimensions],
        "node_count": context.node_count,
        "workload_class": context.workload_class,
        "workspace_budget_bytes": context.workspace_budget_bytes,
    }


def _record_data(
    record: EvidenceRecord | LegacyEvidenceRecord,
) -> dict[str, Any]:
    if type(record) is EvidenceRecord:
        _guard_canonical(record, EvidenceRecord)
        _validate_evidence_record(record)
    elif type(record) is LegacyEvidenceRecord:
        _guard_canonical(record, LegacyEvidenceRecord)
        _validate_legacy_evidence_record(record)
    else:
        raise CompileError("Evidence fingerprint requires a record.")
    _guard_canonical(record.key, EvidenceKey)
    metrics = record.metrics
    if type(metrics) is EvidenceMetrics:
        _guard_canonical(metrics, EvidenceMetrics)
    else:
        _guard_canonical(metrics, LegacyEvidenceMetrics)
    metric_data = {
        "communication_gain_percent": metrics.communication_gain_percent,
        "convergence_step_increase_percent": (
            metrics.convergence_step_increase_percent
        ),
        "cross_workload_reproduced": metrics.cross_workload_reproduced,
        "end_to_end_gain_percent": metrics.end_to_end_gain_percent,
        "quality_loss_percent": metrics.quality_loss_percent,
        "seeds": metrics.seeds,
        "worst_run_gain_percent": metrics.worst_run_gain_percent,
    }
    if record.key.schema_version == EVIDENCE_SCHEMA_VERSION:
        metric_data["exposed_communication_gain_percent"] = (
            metrics.exposed_communication_gain_percent
        )
    return {
        "key": {
            "dimensions": [list(item) for item in record.key.dimensions],
            "schema_version": record.key.schema_version,
        },
        "metrics": metric_data,
        "status": record.status.value,
        "strategy": _legacy_evidence_strategy_data(record.strategy),
    }


def _record_fingerprint(
    record: EvidenceRecord | LegacyEvidenceRecord,
) -> str:
    return _fingerprint(_record_data(record))


def _evidence_generation(evidence: EvidenceStore) -> str:
    records = sorted(
        (
            _record_data(record)
            for record in _valid_evidence_records(evidence)
        ),
        key=lambda value: json.dumps(value, sort_keys=True),
    )
    return _fingerprint(records)


def _plan_signature(
    *,
    intent: CommunicationIntent,
    strategy: StrategySpec,
    context: CompilationContext,
    backend_id: str,
    origin: PlanOrigin,
    evidence_fingerprint: str | None,
) -> str:
    return _fingerprint(
        {
            "backend_id": backend_id,
            "context": _context_data(context),
            "evidence_fingerprint": evidence_fingerprint,
            "intent": _intent_data(intent),
            "origin": origin.value,
            "strategy": _canonical_strategy_data(strategy),
        }
    )
