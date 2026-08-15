"""Evidence-gated compiler orchestration and conservative fallback."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from lowbit_comm.api.intent import (
    CommunicationIntent,
    OutputSemantics,
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
)
from lowbit_comm.backends.protocols import Backend, BackendCapability
from lowbit_comm.compiler.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceKey,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
    LegacyEvidenceRecord,
    _validate_evidence_record,
    _validate_legacy_evidence_record,
)
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import CapabilityError, CompileError
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
)
from lowbit_comm.core.signatures import strategy_key


Policy = NativePolicy | AutoPolicy | ExplicitPolicy
CacheKey = tuple[str, str, str, str, str]


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
        self._cache: dict[CacheKey, ExecutionPlan] = {}

    def compile(
        self,
        intent: CommunicationIntent,
        policy: Policy,
        context: CompilationContext,
    ) -> ExecutionPlan:
        """Resolve, validate, lower, and cache one exact compile request."""
        _validate_compile_inputs(intent, policy, context)
        evidence_generation = _evidence_generation(self._evidence)
        cache_key = (
            _fingerprint(_intent_data(intent)),
            _fingerprint(_policy_data(policy)),
            _fingerprint(_context_data(context)),
            str(self._registry.generation),
            evidence_generation,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        strategy, origin, evidence_record = self._resolve(
            intent,
            policy,
            context,
        )
        _validate_strategy_context(intent, strategy, context)
        capability, backend = _resolve_backend(
            self._registry,
            intent,
            strategy,
        )
        backend_plan = backend.lower(intent, strategy)
        evidence_fingerprint = (
            None
            if evidence_record is None
            else _record_fingerprint(evidence_record)
        )
        signature = _plan_signature(
            intent=intent,
            strategy=strategy,
            context=context,
            backend_id=capability.backend_id,
            origin=origin,
            evidence_fingerprint=evidence_fingerprint,
        )
        plan = ExecutionPlan(
            intent=intent,
            strategy=strategy,
            backend_id=capability.backend_id,
            backend_plan=backend_plan,
            origin=origin,
            signature=signature,
            evidence_fingerprint=evidence_fingerprint,
        )
        self._cache[cache_key] = plan
        return plan

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


def _resolve_backend(
    registry: BackendRegistry,
    intent: CommunicationIntent,
    strategy: StrategySpec,
) -> tuple[BackendCapability, Backend]:
    candidates = registry.candidates(intent, strategy)
    if not candidates:
        raise CapabilityError(
            "No backend supports the exact intent and strategy."
        )
    return candidates[0]


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
    for record in evidence.records:
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
    if type(intent) is not CommunicationIntent:
        raise CompileError("Compiler intent must be CommunicationIntent.")
    if type(policy) not in (NativePolicy, AutoPolicy, ExplicitPolicy):
        raise CompileError("Compiler policy has an unsupported type.")
    if type(context) is not CompilationContext:
        raise CompileError("Compiler context must be CompilationContext.")


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


def _intent_data(intent: CommunicationIntent) -> dict[str, Any]:
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
    return {
        "canonical": [list(component) for component in strategy_key(strategy)]
    }


def _legacy_evidence_strategy_data(
    strategy: StrategySpec,
) -> dict[str, Any]:
    """Return the schema-v1 strategy encoding used by evidence hashes."""
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
        return {"kind": "native"}
    if type(policy) is ExplicitPolicy:
        return {
            "kind": "explicit",
            "strategy": _canonical_strategy_data(policy.strategy),
        }
    return {
        "constraints": _constraints_data(policy.constraints),
        "kind": "auto",
    }


def _context_data(context: CompilationContext) -> dict[str, Any]:
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
        _validate_evidence_record(record)
    elif type(record) is LegacyEvidenceRecord:
        _validate_legacy_evidence_record(record)
    else:
        raise CompileError("Evidence fingerprint requires a record.")
    metrics = record.metrics
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
