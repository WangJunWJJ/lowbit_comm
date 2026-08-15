from dataclasses import FrozenInstanceError, dataclass, fields, replace

import pytest

import lowbit_comm.compiler.compiler as compiler_module
from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
    ShapeFamily,
    TensorSpec,
)
from lowbit_comm.api.policy import (
    AccumulationDType,
    AutoConstraints,
    AutoPolicy,
    CollectiveKind,
    CompressionKind,
    ExplicitPolicy,
    NativePolicy,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.backends.protocols import BackendCapability
from lowbit_comm.compiler.compiler import Compiler
from lowbit_comm.compiler.evidence import (
    EnvironmentFingerprint,
    EvidenceKey,
    EvidenceMetrics,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceStore,
)
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import CapabilityError, CompileError
from lowbit_comm.core.plan import (
    CompilationContext,
    ExecutionPlan,
    PlanOrigin,
)


def _return_value(value: object) -> object:
    return value


@dataclass(frozen=True, slots=True)
class FakeBackendPlan:
    backend_id: str

    def execute(self, value: object) -> object:
        raise AssertionError("execution is outside compiler unit tests")


class StaticMethodBackendPlan:
    """Backend plan exposing execute as a static method."""

    execute = staticmethod(_return_value)


class ClassMethodBackendPlan:
    """Backend plan exposing execute as a class method."""

    @classmethod
    def execute(cls, value: object) -> object:
        del cls
        return value


class InstanceCallableBackendPlan:
    """Backend plan exposing execute as an instance callable field."""

    def __init__(self) -> None:
        self.execute = _return_value


class SlottedCallableBackendPlan:
    """Backend plan storing a callable execute in an instance slot."""

    __slots__ = ("execute",)

    def __init__(self) -> None:
        self.execute = _return_value


class SlottedNonCallableBackendPlan:
    """Backend plan storing a non-callable execute in an instance slot."""

    __slots__ = ("execute",)

    def __init__(self) -> None:
        self.execute = object()


class SlottedUninitializedBackendPlan:
    """Backend plan leaving its execute instance slot uninitialized."""

    __slots__ = ("execute",)


class RaisingPropertyBackendPlan:
    """Backend plan whose execute descriptor must never be evaluated."""

    def __init__(self) -> None:
        self.execute_accesses = 0

    @property
    def execute(self) -> object:
        self.execute_accesses += 1
        raise ValueError("execute property evaluated")


class DynamicExecuteBackendPlan:
    """Backend plan that only pretends to expose execute dynamically."""

    def __init__(self) -> None:
        self.getattr_calls = 0

    def __getattr__(self, name: str) -> object:
        self.getattr_calls += 1
        if name == "execute":
            return _return_value
        raise AttributeError(name)


class NonCallableExecuteBackendPlan:
    """Backend plan exposing a structurally invalid execute field."""

    execute = object()


class FakeBackend:
    def __init__(
        self,
        backend_id: str,
        capability: BackendCapability,
    ) -> None:
        self.backend_id = backend_id
        self._capability = capability
        self.lower_calls = 0

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (self._capability,)

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        if type(strategy) is not StrategySpec:
            raise AssertionError("backend received a non-exact strategy")
        if strategy != self._capability.strategy:
            raise AssertionError("backend received an unsupported strategy")
        self.lower_calls += 1
        return FakeBackendPlan(self.backend_id)


class InvalidPlanBackend(FakeBackend):
    """Backend double that violates the lowering result contract."""

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> object:
        del intent, strategy
        self.lower_calls += 1
        return object()


@dataclass(frozen=True, slots=True)
class CompilerCase:
    registry: BackendRegistry
    native_registry: BackendRegistry
    evidence: EvidenceStore
    production_evidence: EvidenceStore
    intent: CommunicationIntent
    explicit_policy: ExplicitPolicy
    auto_policy: AutoPolicy
    context: CompilationContext


def _metrics() -> EvidenceMetrics:
    return EvidenceMetrics(
        communication_gain_percent=12.0,
        end_to_end_gain_percent=10.0,
        quality_loss_percent=0.5,
        convergence_step_increase_percent=2.0,
        worst_run_gain_percent=-2.0,
        seeds=3,
        cross_workload_reproduced=True,
    )


def exact_compressed_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        error_feedback=True,
        overlap=True,
    )


def compiler_case() -> CompilerCase:
    intent = CommunicationIntent(
        tensor=TensorSpec(dtype="float16", shape=(1024,)),
        shape_family=ShapeFamily(max_numel=1024, alignment=2),
        reduction=ReductionOp.MEAN,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=4,
        rank=0,
    )
    explicit_strategy = exact_compressed_strategy()
    native_strategy = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    compressed_capability = BackendCapability(
        backend_id="cuda",
        strategy=explicit_strategy,
        output=intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    native_capability = BackendCapability(
        backend_id="native",
        strategy=native_strategy,
        output=intent.output,
        min_world_size=2,
        max_world_size=None,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    registry = BackendRegistry(
        [
            FakeBackend("cuda", compressed_capability),
            FakeBackend("native", native_capability),
        ]
    )
    native_registry = BackendRegistry(
        [FakeBackend("native", native_capability)]
    )
    context = CompilationContext(
        environment=EnvironmentFingerprint.from_mapping(
            {
                "hardware": "a6000",
                "interconnect": "pcie4",
                "software": "test",
            }
        ),
        workspace_budget_bytes=256 * 1024 * 1024,
        node_count=1,
        workload_class="communication_bound",
        bucket_min_bytes=2048,
        bucket_max_bytes=2048,
    )
    production_record = EvidenceRecord(
        key=EvidenceKey.from_request(
            environment=context.environment,
            intent=intent,
            strategy=explicit_strategy,
            node_count=context.node_count,
            workload_class=context.workload_class,
            bucket_min_bytes=context.bucket_min_bytes,
            bucket_max_bytes=context.bucket_max_bytes,
        ),
        strategy=explicit_strategy,
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=_metrics(),
    )
    return CompilerCase(
        registry=registry,
        native_registry=native_registry,
        evidence=EvidenceStore(),
        production_evidence=EvidenceStore([production_record]),
        intent=intent,
        explicit_policy=ExplicitPolicy(explicit_strategy),
        auto_policy=AutoPolicy(),
        context=context,
    )


def execution_plan(
    case: CompilerCase,
    backend_plan: object,
) -> ExecutionPlan:
    return ExecutionPlan(
        intent=case.intent,
        strategy=case.explicit_policy.strategy,
        backend_id="cuda",
        backend_plan=backend_plan,  # type: ignore[arg-type]
        origin=PlanOrigin.EXPLICIT,
        signature="descriptor-validation",
        evidence_fingerprint=None,
    )


def evidence_for(
    case: CompilerCase,
    strategy: StrategySpec,
    status: EvidenceStatus = EvidenceStatus.PRODUCTION_AUTO,
) -> EvidenceRecord:
    return EvidenceRecord(
        key=EvidenceKey.from_request(
            environment=case.context.environment,
            intent=case.intent,
            strategy=strategy,
            node_count=case.context.node_count,
            workload_class=case.context.workload_class,
            bucket_min_bytes=case.context.bucket_min_bytes,
            bucket_max_bytes=case.context.bucket_max_bytes,
        ),
        strategy=strategy,
        status=status,
        metrics=_metrics(),
    )


def unsupported_capability_strategies(
    strategy: StrategySpec,
) -> tuple[tuple[str, StrategySpec], ...]:
    native = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    return (
        ("compression", native),
        ("collective", native),
        ("topology", replace(strategy, topology=TopologyKind.TREE)),
        ("group_size", replace(strategy, group_size=64)),
        (
            "accumulation_dtype",
            replace(
                strategy,
                accumulation_dtype=AccumulationDType.FP16,
            ),
        ),
        (
            "error_feedback",
            replace(strategy, error_feedback=not strategy.error_feedback),
        ),
        (
            "parameter_error_feedback",
            replace(strategy, parameter_error_feedback=True),
        ),
        ("overlap", replace(strategy, overlap=not strategy.overlap)),
        (
            "workspace_budget_bytes",
            replace(strategy, workspace_budget_bytes=1024),
        ),
    )


def test_compiler_cases_cover_every_strategy_field() -> None:
    assert {
        name
        for name, _ in unsupported_capability_strategies(
            exact_compressed_strategy()
        )
    } == {field.name for field in fields(StrategySpec)}


def test_native_policy_compiles_same_output_native_capability() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.evidence).compile(
        case.intent,
        NativePolicy(),
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE
    assert plan.strategy.compression is CompressionKind.NONE


def test_explicit_strategy_compiles_exact_capability() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.evidence).compile(
        case.intent,
        case.explicit_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.EXPLICIT
    assert plan.strategy == case.explicit_policy.strategy


def test_explicit_strategy_never_falls_back() -> None:
    case = compiler_case()

    with pytest.raises(CapabilityError):
        Compiler(case.native_registry, case.evidence).compile(
            case.intent,
            case.explicit_policy,
            case.context,
        )


@pytest.mark.parametrize(
    ("field", "advertised_strategy"),
    unsupported_capability_strategies(exact_compressed_strategy()),
)
def test_explicit_strategy_difference_raises_before_lowering(
    field: str,
    advertised_strategy: StrategySpec,
) -> None:
    case = compiler_case()
    requested = case.explicit_policy.strategy
    capability = BackendCapability(
        backend_id="unsupported",
        strategy=advertised_strategy,
        output=case.intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    backend = FakeBackend("unsupported", capability)

    assert getattr(advertised_strategy, field) != getattr(requested, field)
    with pytest.raises(CapabilityError):
        Compiler(
            BackendRegistry([backend]),
            EvidenceStore(),
        ).compile(
            case.intent,
            case.explicit_policy,
            case.context,
        )
    assert backend.lower_calls == 0


def test_auto_without_exact_production_evidence_compiles_native() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


@pytest.mark.parametrize(
    ("field", "advertised_strategy"),
    unsupported_capability_strategies(exact_compressed_strategy()),
)
def test_auto_strategy_difference_falls_back_without_compressed_lowering(
    field: str,
    advertised_strategy: StrategySpec,
) -> None:
    case = compiler_case()
    requested = case.explicit_policy.strategy
    native_strategy = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    native_capability = case.registry.candidates(
        case.intent, native_strategy
    )[0][0]
    unsupported_capability = BackendCapability(
        backend_id="unsupported",
        strategy=advertised_strategy,
        output=case.intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    native_backend = FakeBackend("native", native_capability)
    unsupported_backend = FakeBackend(
        "unsupported", unsupported_capability
    )

    plan = Compiler(
        BackendRegistry([unsupported_backend, native_backend]),
        case.production_evidence,
    ).compile(case.intent, case.auto_policy, case.context)

    assert getattr(advertised_strategy, field) != getattr(requested, field)
    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert unsupported_backend.lower_calls == 0
    assert native_backend.lower_calls == 1


def test_auto_plan_is_immutable_and_cached() -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.evidence)

    first = compiler.compile(case.intent, case.auto_policy, case.context)
    second = compiler.compile(case.intent, case.auto_policy, case.context)

    assert first is second
    with pytest.raises(FrozenInstanceError):
        first.signature = "changed"


def test_auto_with_exact_production_evidence_selects_compressed_plan() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.AUTO
    assert plan.strategy == case.explicit_policy.strategy


def test_native_policy_has_priority_over_production_evidence() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        NativePolicy(),
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE


def test_explicit_policy_has_priority_over_production_evidence() -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.explicit_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.EXPLICIT


def test_recommended_evidence_cannot_drive_auto() -> None:
    case = compiler_case()
    recommended = evidence_for(
        case,
        case.explicit_policy.strategy,
        EvidenceStatus.RECOMMENDED,
    )

    plan = Compiler(case.registry, EvidenceStore([recommended])).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_constrained_auto_filters_the_evidence_selected_strategy() -> None:
    case = compiler_case()
    policy = AutoPolicy(
        AutoConstraints(
            denied_compressions=frozenset({CompressionKind.INT8}),
        )
    )

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


@pytest.mark.parametrize(
    "constraints",
    [
        AutoConstraints(
            allowed_compressions=frozenset({CompressionKind.NONE}),
        ),
        AutoConstraints(
            allowed_collectives=frozenset({CollectiveKind.NATIVE}),
        ),
        AutoConstraints(
            denied_collectives=frozenset(
                {CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE}
            ),
        ),
        AutoConstraints(
            allowed_topologies=frozenset({TopologyKind.TREE}),
        ),
        AutoConstraints(
            denied_topologies=frozenset({TopologyKind.RING}),
        ),
    ],
)
def test_all_auto_constraint_dimensions_filter_selected_evidence(
    constraints: AutoConstraints,
) -> None:
    case = compiler_case()

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        AutoPolicy(constraints),
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_auto_workspace_constraint_filters_selected_evidence() -> None:
    case = compiler_case()
    strategy = replace(
        case.explicit_policy.strategy,
        workspace_budget_bytes=1024,
    )
    evidence = EvidenceStore([evidence_for(case, strategy)])
    policy = AutoPolicy(AutoConstraints(max_workspace_bytes=512))

    plan = Compiler(case.registry, evidence).compile(
        case.intent,
        policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_constraints_do_not_search_for_another_evidence_strategy() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    tree_capability = BackendCapability(
        backend_id="tree",
        strategy=tree,
        output=case.intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    case.registry.register(FakeBackend("tree", tree_capability))
    evidence = EvidenceStore(
        [evidence_for(case, tree), evidence_for(case, ring)]
    )
    policy = AutoPolicy(
        AutoConstraints(
            allowed_topologies=frozenset({TopologyKind.TREE}),
        )
    )

    plan = Compiler(case.registry, evidence).compile(
        case.intent,
        policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_compiler_never_calls_diagnostic_capability_enumeration() -> None:
    case = compiler_case()

    def forbidden(world_size: int) -> object:
        raise AssertionError("diagnostic lookup must not select strategies")

    setattr(case.registry, "capabilities_for_world_size", forbidden)

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.AUTO


def test_auto_uses_the_typed_evidence_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = compiler_case()

    def forbidden(key: EvidenceKey) -> StrategySpec:
        raise AssertionError("EvidenceKey strings must not be decoded")

    monkeypatch.setattr(
        compiler_module,
        "_strategy_from_evidence_key",
        forbidden,
        raising=False,
    )

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.AUTO


def test_auto_with_unavailable_evidence_strategy_falls_back_native() -> None:
    case = compiler_case()

    plan = Compiler(
        case.native_registry,
        case.production_evidence,
    ).compile(case.intent, case.auto_policy, case.context)

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_auto_requires_an_exact_evidence_context_match() -> None:
    case = compiler_case()
    changed_context = replace(case.context, bucket_max_bytes=4096)

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        changed_context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK


def test_local_rank_does_not_change_auto_selection() -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.production_evidence)
    rank_zero = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )
    rank_one = compiler.compile(
        replace(case.intent, rank=1),
        case.auto_policy,
        case.context,
    )

    assert rank_zero.origin is rank_one.origin is PlanOrigin.AUTO
    assert rank_zero.strategy == rank_one.strategy
    assert rank_zero.evidence_fingerprint == rank_one.evidence_fingerprint
    assert rank_zero.signature != rank_one.signature


def test_cache_keys_use_values_instead_of_object_identity() -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.production_evidence)
    first = compiler.compile(case.intent, case.auto_policy, case.context)

    second = compiler.compile(
        replace(case.intent),
        AutoPolicy(),
        replace(case.context),
    )

    assert first is second


def test_plan_signature_excludes_backend_plan_identity() -> None:
    first_case = compiler_case()
    second_case = compiler_case()
    first = Compiler(
        first_case.registry,
        first_case.production_evidence,
    ).compile(
        first_case.intent,
        first_case.auto_policy,
        first_case.context,
    )
    second = Compiler(
        second_case.registry,
        second_case.production_evidence,
    ).compile(
        second_case.intent,
        second_case.auto_policy,
        second_case.context,
    )

    assert first.backend_plan is not second.backend_plan
    assert first.signature == second.signature


def test_schema_one_evidence_fingerprint_remains_legacy_stable() -> None:
    record = compiler_case().production_evidence.records[0]

    assert compiler_module._record_fingerprint(record) == (
        "535046cd702cc06eb66b24ca1ab83b0f3a9a53407e31be7db9b5909bd435484a"
    )


@pytest.mark.parametrize(
    "backend_plan",
    [
        FakeBackendPlan("cuda"),
        StaticMethodBackendPlan(),
        ClassMethodBackendPlan(),
        InstanceCallableBackendPlan(),
    ],
)
def test_execution_plan_accepts_static_callable_execute(
    backend_plan: object,
) -> None:
    case = compiler_case()

    plan = execution_plan(case, backend_plan)

    assert plan.backend_plan is backend_plan


def test_execution_plan_accepts_slotted_callable_execute() -> None:
    case = compiler_case()
    backend_plan = SlottedCallableBackendPlan()

    plan = execution_plan(case, backend_plan)

    assert plan.backend_plan is backend_plan


@pytest.mark.parametrize(
    "backend_plan",
    [
        SlottedNonCallableBackendPlan(),
        SlottedUninitializedBackendPlan(),
    ],
)
def test_execution_plan_rejects_invalid_slotted_execute(
    backend_plan: object,
) -> None:
    case = compiler_case()

    with pytest.raises(CompileError, match="execute"):
        execution_plan(case, backend_plan)


def test_execution_plan_rejects_execute_property_without_accessing_it(
) -> None:
    case = compiler_case()
    backend_plan = RaisingPropertyBackendPlan()

    with pytest.raises(CompileError, match="execute"):
        execution_plan(case, backend_plan)

    assert backend_plan.execute_accesses == 0


def test_execution_plan_rejects_dynamic_execute_without_lookup() -> None:
    case = compiler_case()
    backend_plan = DynamicExecuteBackendPlan()

    with pytest.raises(CompileError, match="execute"):
        execution_plan(case, backend_plan)

    assert backend_plan.getattr_calls == 0


def test_execution_plan_rejects_non_callable_execute() -> None:
    case = compiler_case()

    with pytest.raises(CompileError, match="execute"):
        execution_plan(case, NonCallableExecuteBackendPlan())


def test_invalid_lowered_plan_is_rejected_and_never_cached() -> None:
    case = compiler_case()
    strategy = case.explicit_policy.strategy
    capability, _ = case.registry.candidates(case.intent, strategy)[0]
    backend = InvalidPlanBackend("cuda", capability)
    compiler = Compiler(
        BackendRegistry([backend]),
        EvidenceStore(),
    )

    for _ in range(2):
        with pytest.raises(CompileError, match="execute"):
            compiler.compile(
                case.intent,
                case.explicit_policy,
                case.context,
            )

    assert backend.lower_calls == 2


def test_registry_generation_invalidates_a_cached_fallback() -> None:
    case = compiler_case()
    compiler = Compiler(case.native_registry, case.production_evidence)
    fallback = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )
    capability, _ = case.registry.candidates(
        case.intent,
        case.explicit_policy.strategy,
    )[0]

    case.native_registry.register(FakeBackend("cuda", capability))
    selected = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert fallback.origin is PlanOrigin.NATIVE_FALLBACK
    assert selected.origin is PlanOrigin.AUTO
    assert selected is not fallback


def test_native_fallback_preserves_reduced_shard_output() -> None:
    case = compiler_case()
    intent = replace(case.intent, output=OutputSemantics.REDUCED_SHARD)
    native = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    capability = BackendCapability(
        backend_id="native",
        strategy=native,
        output=intent.output,
        min_world_size=2,
        max_world_size=None,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    registry = BackendRegistry([FakeBackend("native", capability)])

    plan = Compiler(registry, EvidenceStore()).compile(
        intent,
        AutoPolicy(),
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert capability.output is OutputSemantics.REDUCED_SHARD


@pytest.mark.parametrize(
    "strategy",
    [
        StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            topology=TopologyKind.RING,
            group_size=128,
            workspace_budget_bytes=1024,
        ),
        StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            topology=TopologyKind.RING,
            group_size=128,
            parameter_error_feedback=True,
        ),
    ],
)
def test_invalid_strategy_context_is_rejected_before_lowering(
    strategy: StrategySpec,
) -> None:
    case = compiler_case()
    capability = BackendCapability(
        backend_id="cuda",
        strategy=strategy,
        output=case.intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    backend = FakeBackend("cuda", capability)
    context = replace(case.context, workspace_budget_bytes=512)

    with pytest.raises(CompileError):
        Compiler(BackendRegistry([backend]), EvidenceStore()).compile(
            case.intent,
            ExplicitPolicy(strategy),
            context,
        )

    assert backend.lower_calls == 0


def test_invalid_output_strategy_is_rejected_before_lowering() -> None:
    case = compiler_case()
    intent = replace(case.intent, output=OutputSemantics.REDUCED_SHARD)
    strategy = case.explicit_policy.strategy
    capability = BackendCapability(
        backend_id="cuda",
        strategy=strategy,
        output=intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )
    backend = FakeBackend("cuda", capability)

    with pytest.raises(CompileError):
        Compiler(BackendRegistry([backend]), EvidenceStore()).compile(
            intent,
            ExplicitPolicy(strategy),
            case.context,
        )

    assert backend.lower_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_budget_bytes", True),
        ("node_count", True),
        ("workload_class", ""),
        ("bucket_min_bytes", -1),
        ("bucket_max_bytes", True),
    ],
)
def test_compilation_context_rejects_invalid_signature_fields(
    field: str,
    value: object,
) -> None:
    case = compiler_case()
    values: dict[str, object] = {
        "environment": case.context.environment,
        "workspace_budget_bytes": case.context.workspace_budget_bytes,
        "node_count": case.context.node_count,
        "workload_class": case.context.workload_class,
        "bucket_min_bytes": case.context.bucket_min_bytes,
        "bucket_max_bytes": case.context.bucket_max_bytes,
    }
    values[field] = value

    with pytest.raises(CompileError):
        CompilationContext(**values)  # type: ignore[arg-type]
