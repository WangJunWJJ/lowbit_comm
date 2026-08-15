from dataclasses import FrozenInstanceError, dataclass, fields, replace

import pytest

import lowbit_comm
import lowbit_comm.compiler.compiler as compiler_module
from lowbit_comm.api.communicator import compile_communicator
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
    LegacyEvidenceMetrics,
    LegacyEvidenceRecord,
)
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
    ExecutionError,
    LowbitCommError,
)
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


class DynamicLowerTrapBackend(FakeBackend):
    """Expose a valid class lower behind malicious dynamic lookup."""

    def __init__(
        self,
        backend_id: str,
        capability: BackendCapability,
        trap: str,
    ) -> None:
        super().__init__(backend_id, capability)
        self.trap = trap
        self.lower_accesses = 0
        self.trap_calls = 0

    def __getattribute__(self, name: str) -> object:
        if name == "lower":
            accesses = object.__getattribute__(self, "lower_accesses")
            object.__setattr__(self, "lower_accesses", accesses + 1)
            if object.__getattribute__(self, "trap") == "raise":
                raise RuntimeError("dynamic lower lookup executed")

            def wrong_lower(
                intent: CommunicationIntent,
                strategy: StrategySpec,
            ) -> object:
                del intent, strategy
                calls = object.__getattribute__(self, "trap_calls")
                object.__setattr__(self, "trap_calls", calls + 1)
                return object()

            return wrong_lower
        return object.__getattribute__(self, name)


class MultiRegistrationLowerTrapBackend:
    """Advertise successive batches behind one statically bound lower."""

    def __init__(
        self,
        backend_id: str,
        capabilities: tuple[BackendCapability, ...],
    ) -> None:
        self.backend_id = backend_id
        self._current_capabilities = capabilities
        self._supported_strategies = tuple(
            capability.strategy for capability in capabilities
        )
        self.capabilities_calls = 0
        self.lower_accesses = 0
        self.lower_calls = 0

    def advertise(
        self,
        capabilities: tuple[BackendCapability, ...],
    ) -> None:
        self._current_capabilities = capabilities
        self._supported_strategies += tuple(
            capability.strategy for capability in capabilities
        )

    def capabilities(self) -> tuple[BackendCapability, ...]:
        self.capabilities_calls += 1
        return self._current_capabilities

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        del intent
        if strategy not in self._supported_strategies:
            raise AssertionError("backend received an unsupported strategy")
        self.lower_calls += 1
        return FakeBackendPlan(self.backend_id)

    def __getattribute__(self, name: str) -> object:
        if name == "lower":
            accesses = object.__getattribute__(self, "lower_accesses")
            object.__setattr__(self, "lower_accesses", accesses + 1)
            raise RuntimeError("dynamic lower lookup executed")
        return object.__getattribute__(self, name)


class CountingLower:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.calls = 0

    def __call__(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        del intent, strategy
        self.calls += 1
        return FakeBackendPlan(self.backend_id)


class StaticLowerBackend(FakeBackend):
    lower_calls = 0

    @staticmethod
    def lower(
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        del intent, strategy
        StaticLowerBackend.lower_calls += 1
        return FakeBackendPlan("cuda")


class ClassLowerBackend(FakeBackend):
    lower_calls = 0

    @classmethod
    def lower(
        cls,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        del intent, strategy
        cls.lower_calls += 1
        return FakeBackendPlan("cuda")


class InstanceCallableLowerBackend(FakeBackend):
    def __init__(
        self,
        backend_id: str,
        capability: BackendCapability,
    ) -> None:
        super().__init__(backend_id, capability)
        self.lower = CountingLower(backend_id)  # type: ignore[method-assign]


class SlottedCallableLowerBackend:
    __slots__ = ("backend_id", "_capability", "lower")

    def __init__(
        self,
        backend_id: str,
        capability: BackendCapability,
    ) -> None:
        self.backend_id = backend_id
        self._capability = capability
        self.lower = CountingLower(backend_id)

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (self._capability,)


class RaisingLowerBackend(FakeBackend):
    def __init__(
        self,
        backend_id: str,
        capability: BackendCapability,
        error: Exception,
    ) -> None:
        super().__init__(backend_id, capability)
        self.error = error

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> FakeBackendPlan:
        del intent, strategy
        self.lower_calls += 1
        raise self.error


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
        exposed_communication_gain_percent=1.0,
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


def exact_native_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
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
    *,
    intent: CommunicationIntent | None = None,
) -> EvidenceRecord:
    evidence_intent = case.intent if intent is None else intent
    return EvidenceRecord(
        key=EvidenceKey.from_request(
            environment=case.context.environment,
            intent=evidence_intent,
            strategy=strategy,
            node_count=case.context.node_count,
            workload_class=case.context.workload_class,
            bucket_min_bytes=case.context.bucket_min_bytes,
            bucket_max_bytes=case.context.bucket_max_bytes,
        ),
        strategy=strategy,
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=_metrics(),
    )


def backend_for_strategy(
    case: CompilerCase,
    backend_id: str,
    strategy: StrategySpec,
) -> FakeBackend:
    return FakeBackend(
        backend_id,
        capability_for_strategy(case, backend_id, strategy),
    )


def capability_for_strategy(
    case: CompilerCase,
    backend_id: str,
    strategy: StrategySpec,
) -> BackendCapability:
    return BackendCapability(
        backend_id=backend_id,
        strategy=strategy,
        output=case.intent.output,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
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


def test_compiler_canonical_classifiers_cover_exact_dataclass_fields() -> None:
    assert compiler_module._CANONICAL_DATACLASS_FIELDS == {
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
    for contract_type, classified in (
        compiler_module._CANONICAL_DATACLASS_FIELDS.items()
    ):
        assert classified == frozenset(
            field.name for field in fields(contract_type)
        )


def _legacy_record_for_canonical_test(
    case: CompilerCase,
) -> LegacyEvidenceRecord:
    current = case.production_evidence.records[0]
    return LegacyEvidenceRecord(
        key=EvidenceKey.from_mapping(
            schema_version=1,
            dimensions=dict(current.key.dimensions),
        ),
        strategy=current.strategy,
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=LegacyEvidenceMetrics(
            communication_gain_percent=12.0,
            end_to_end_gain_percent=10.0,
            quality_loss_percent=0.5,
            convergence_step_increase_percent=2.0,
            worst_run_gain_percent=-2.0,
            seeds=3,
            cross_workload_reproduced=True,
        ),
    )


def _canonicalize_contract_for_test(
    case: CompilerCase,
    contract_type: type[object],
) -> object:
    if contract_type in (CommunicationIntent, TensorSpec, ShapeFamily):
        return compiler_module._intent_data(case.intent)
    if contract_type is StrategySpec:
        return compiler_module._canonical_strategy_data(
            case.explicit_policy.strategy
        )
    if contract_type is AutoConstraints:
        return compiler_module._constraints_data(
            case.auto_policy.constraints
        )
    if contract_type is NativePolicy:
        return compiler_module._policy_data(NativePolicy())
    if contract_type is AutoPolicy:
        return compiler_module._policy_data(case.auto_policy)
    if contract_type is ExplicitPolicy:
        return compiler_module._policy_data(case.explicit_policy)
    if contract_type in (EnvironmentFingerprint, CompilationContext):
        return compiler_module._context_data(case.context)
    if contract_type in (
        EvidenceKey,
        EvidenceMetrics,
        EvidenceRecord,
    ):
        return compiler_module._record_data(
            case.production_evidence.records[0]
        )
    return compiler_module._record_data(
        _legacy_record_for_canonical_test(case)
    )


@pytest.mark.parametrize(
    "contract_type",
    [
        CommunicationIntent,
        TensorSpec,
        ShapeFamily,
        StrategySpec,
        AutoConstraints,
        NativePolicy,
        AutoPolicy,
        ExplicitPolicy,
        EnvironmentFingerprint,
        CompilationContext,
        EvidenceKey,
        LegacyEvidenceMetrics,
        EvidenceMetrics,
        LegacyEvidenceRecord,
        EvidenceRecord,
    ],
)
def test_each_canonicalizer_fails_closed_on_classifier_drift(
    contract_type: type[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifiers = dict(compiler_module._CANONICAL_DATACLASS_FIELDS)
    current = classifiers[contract_type]
    classifiers[contract_type] = (
        frozenset({"future_field"})
        if not current
        else frozenset(tuple(current)[1:])
    )
    monkeypatch.setattr(
        compiler_module,
        "_CANONICAL_DATACLASS_FIELDS",
        classifiers,
    )

    with pytest.raises(CompileError, match="canonical fields"):
        _canonicalize_contract_for_test(compiler_case(), contract_type)


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


@pytest.mark.parametrize("trap", ["raise", "wrong_callable"])
@pytest.mark.parametrize(
    ("policy_path", "expected_origin"),
    [
        ("native", PlanOrigin.NATIVE),
        ("explicit", PlanOrigin.EXPLICIT),
        ("auto", PlanOrigin.AUTO),
        ("auto_fallback", PlanOrigin.NATIVE_FALLBACK),
    ],
)
def test_compiler_uses_registered_lower_without_dynamic_access(
    trap: str,
    policy_path: str,
    expected_origin: PlanOrigin,
) -> None:
    case = compiler_case()
    if policy_path in ("native", "auto_fallback"):
        backend_id = "native"
        strategy = exact_native_strategy()
    else:
        backend_id = "cuda"
        strategy = exact_compressed_strategy()
    capability = capability_for_strategy(
        case,
        backend_id,
        strategy,
    )
    backend = DynamicLowerTrapBackend(backend_id, capability, trap)
    registry = BackendRegistry([backend])
    assert backend.lower_accesses == 0
    if policy_path == "native":
        policy: NativePolicy | ExplicitPolicy | AutoPolicy = NativePolicy()
        evidence = case.evidence
    elif policy_path == "explicit":
        policy = ExplicitPolicy(strategy)
        evidence = case.evidence
    elif policy_path == "auto":
        policy = AutoPolicy()
        evidence = case.production_evidence
    else:
        policy = AutoPolicy()
        evidence = case.evidence
    compiler = Compiler(registry, evidence)

    plan = compiler.compile(case.intent, policy, case.context)
    cached = compiler.compile(case.intent, policy, case.context)

    assert plan.origin is expected_origin
    assert cached is plan
    assert backend.lower_accesses == 0
    assert backend.trap_calls == 0
    assert backend.lower_calls == 1


def test_compiler_preserves_owner_and_bound_lower_across_registrations(
) -> None:
    case = compiler_case()
    ring = exact_compressed_strategy()
    tree = replace(ring, topology=TopologyKind.TREE)
    ring_capability = capability_for_strategy(case, "cuda", ring)
    tree_capability = capability_for_strategy(case, "cuda", tree)
    backend = MultiRegistrationLowerTrapBackend(
        "cuda",
        (ring_capability,),
    )
    registry = BackendRegistry()

    registry.register(backend)
    backend.advertise((tree_capability,))
    registry.register(backend)
    compiler = Compiler(registry, case.evidence)

    ring_plan = compiler.compile(
        case.intent,
        ExplicitPolicy(ring),
        case.context,
    )
    tree_plan = compiler.compile(
        case.intent,
        ExplicitPolicy(tree),
        case.context,
    )
    cached_ring_plan = compiler.compile(
        case.intent,
        ExplicitPolicy(ring),
        case.context,
    )

    assert ring_plan.backend_id == tree_plan.backend_id == "cuda"
    assert cached_ring_plan is ring_plan
    assert all(
        match[1] is backend
        for match in registry.capabilities_for_world_size(4)
    )
    assert registry.generation == 2
    assert backend.capabilities_calls == 2
    assert backend.lower_accesses == 0
    assert backend.lower_calls == 2


@pytest.mark.parametrize(
    "lower_form",
    ["normal", "static", "class", "instance", "slot"],
)
def test_compiler_invokes_every_registered_lower_form(
    lower_form: str,
) -> None:
    case = compiler_case()
    capability = capability_for_strategy(
        case,
        "cuda",
        exact_compressed_strategy(),
    )
    StaticLowerBackend.lower_calls = 0
    ClassLowerBackend.lower_calls = 0
    if lower_form == "normal":
        backend: object = FakeBackend("cuda", capability)
    elif lower_form == "static":
        backend = StaticLowerBackend("cuda", capability)
    elif lower_form == "class":
        backend = ClassLowerBackend("cuda", capability)
    elif lower_form == "instance":
        backend = InstanceCallableLowerBackend("cuda", capability)
    else:
        backend = SlottedCallableLowerBackend("cuda", capability)
    compiler = Compiler(
        BackendRegistry([backend]),  # type: ignore[list-item]
        case.evidence,
    )

    plan = compiler.compile(
        case.intent,
        ExplicitPolicy(capability.strategy),
        case.context,
    )

    assert plan.backend_id == "cuda"
    if lower_form == "normal":
        assert backend.lower_calls == 1  # type: ignore[attr-defined]
    elif lower_form == "static":
        assert StaticLowerBackend.lower_calls == 1
    elif lower_form == "class":
        assert ClassLowerBackend.lower_calls == 1
    else:
        assert backend.lower.calls == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "error",
    [
        LowbitCommError("lowbit failure"),
        CompileError("compile failure"),
        CapabilityError("capability failure"),
        ExecutionError("execution failure"),
    ],
)
def test_lowering_preserves_lowbit_error_identity(
    error: LowbitCommError,
) -> None:
    case = compiler_case()
    capability = capability_for_strategy(
        case,
        "cuda",
        exact_compressed_strategy(),
    )
    backend = RaisingLowerBackend("cuda", capability, error)
    compiler = Compiler(BackendRegistry([backend]), case.evidence)

    with pytest.raises(type(error)) as caught:
        compiler.compile(
            case.intent,
            ExplicitPolicy(capability.strategy),
            case.context,
        )

    assert caught.value is error
    assert backend.lower_calls == 1


def test_lowering_normalizes_unknown_exception() -> None:
    case = compiler_case()
    capability = capability_for_strategy(
        case,
        "cuda",
        exact_compressed_strategy(),
    )
    error = RuntimeError("unknown lowering failure")
    backend = RaisingLowerBackend("cuda", capability, error)
    compiler = Compiler(BackendRegistry([backend]), case.evidence)

    with pytest.raises(
        CompileError,
        match="Backend lower.*failed during compilation",
    ) as caught:
        compiler.compile(
            case.intent,
            ExplicitPolicy(capability.strategy),
            case.context,
        )

    assert caught.value.__cause__ is error
    assert backend.lower_calls == 1


def test_explicit_strategy_never_falls_back() -> None:
    case = compiler_case()

    with pytest.raises(lowbit_comm.CapabilityError):
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
    strategy = case.explicit_policy.strategy
    recommended = EvidenceRecord(
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
        status=EvidenceStatus.RECOMMENDED,
        metrics=EvidenceMetrics(
            communication_gain_percent=12.0,
            exposed_communication_gain_percent=1.0,
            end_to_end_gain_percent=5.0,
            quality_loss_percent=0.5,
            convergence_step_increase_percent=2.0,
            worst_run_gain_percent=-2.0,
            seeds=3,
            cross_workload_reproduced=False,
        ),
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


def test_auto_continues_after_policy_rejects_first_evidence() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    ring_backend = case.registry.candidates(case.intent, ring)[0][1]
    native = exact_native_strategy()
    native_backend = case.registry.candidates(case.intent, native)[0][1]
    tree_backend = backend_for_strategy(case, "tree", tree)
    case.registry.register(tree_backend)
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

    assert plan.origin is PlanOrigin.AUTO
    assert plan.strategy == tree
    assert plan.backend_id == "tree"
    assert ring_backend.lower_calls == 0
    assert tree_backend.lower_calls == 1
    assert native_backend.lower_calls == 0


def test_auto_continues_after_first_evidence_lacks_capability() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    tree_backend = backend_for_strategy(case, "tree", tree)
    case.native_registry.register(tree_backend)
    evidence = EvidenceStore(
        [evidence_for(case, tree), evidence_for(case, ring)]
    )

    plan = Compiler(case.native_registry, evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.AUTO
    assert plan.strategy == tree
    assert tree_backend.lower_calls == 1


def test_auto_continues_after_first_evidence_fails_context() -> None:
    case = compiler_case()
    ring = replace(
        case.explicit_policy.strategy,
        workspace_budget_bytes=case.context.workspace_budget_bytes + 1,
    )
    tree = replace(
        case.explicit_policy.strategy,
        topology=TopologyKind.TREE,
    )
    ring_backend = backend_for_strategy(case, "ring-workspace", ring)
    tree_backend = backend_for_strategy(case, "tree", tree)
    case.registry.register(ring_backend)
    case.registry.register(tree_backend)
    evidence = EvidenceStore(
        [evidence_for(case, tree), evidence_for(case, ring)]
    )

    plan = Compiler(case.registry, evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.AUTO
    assert plan.strategy == tree
    assert ring_backend.lower_calls == 0
    assert tree_backend.lower_calls == 1


def test_auto_evidence_selection_is_independent_of_store_order() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    tree_backend = backend_for_strategy(case, "tree", tree)
    case.registry.register(tree_backend)
    ring_record = evidence_for(case, ring)
    tree_record = evidence_for(case, tree)
    policy = AutoPolicy(
        AutoConstraints(
            allowed_topologies=frozenset({TopologyKind.TREE}),
        )
    )

    forward = Compiler(
        case.registry,
        EvidenceStore([ring_record, tree_record]),
    ).compile(case.intent, policy, case.context)
    reverse = Compiler(
        case.registry,
        EvidenceStore([tree_record, ring_record]),
    ).compile(case.intent, policy, case.context)

    assert forward.strategy == reverse.strategy == tree
    assert forward.backend_id == reverse.backend_id == "tree"
    assert forward.signature == reverse.signature
    assert tree_backend.lower_calls == 2


def test_auto_falls_back_after_all_exact_evidence_is_ineligible() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    native = exact_native_strategy()
    native_backend = case.native_registry.candidates(
        case.intent,
        native,
    )[0][1]
    evidence = EvidenceStore(
        [evidence_for(case, tree), evidence_for(case, ring)]
    )
    policy = AutoPolicy(
        AutoConstraints(
            denied_topologies=frozenset({TopologyKind.RING}),
        )
    )

    plan = Compiler(case.native_registry, evidence).compile(
        case.intent,
        policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert native_backend.lower_calls == 1


def test_forging_selected_evidence_reselects_later_valid_record() -> None:
    case = compiler_case()
    ring = case.explicit_policy.strategy
    tree = replace(ring, topology=TopologyKind.TREE)
    ring_backend = case.registry.candidates(case.intent, ring)[0][1]
    tree_backend = backend_for_strategy(case, "tree", tree)
    case.registry.register(tree_backend)
    ring_record = evidence_for(case, ring)
    tree_record = evidence_for(case, tree)
    evidence = EvidenceStore([tree_record, ring_record])
    compiler = Compiler(case.registry, evidence)

    first = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )
    object.__setattr__(ring_record.metrics, "quality_loss_percent", 5.0)
    second = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert first.origin is second.origin is PlanOrigin.AUTO
    assert first.strategy == ring
    assert second.strategy == tree
    assert second is not first
    assert ring_backend.lower_calls == 1
    assert tree_backend.lower_calls == 1


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


@pytest.mark.parametrize(
    "evidence_intent",
    [
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(32, 32)),
            shape_family=ShapeFamily(max_numel=1024, alignment=2),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(1024,)),
            shape_family=ShapeFamily(max_numel=2048, alignment=2),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(1024,)),
            shape_family=ShapeFamily(max_numel=1024, alignment=2),
            reduction=ReductionOp.SUM,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
            world_size=4,
            rank=0,
        ),
        CommunicationIntent(
            tensor=TensorSpec(dtype="float16", shape=(1024,)),
            shape_family=ShapeFamily(max_numel=1024, alignment=2),
            reduction=ReductionOp.MEAN,
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.SYNC,
            world_size=4,
            rank=0,
        ),
    ],
)
def test_auto_falls_back_for_different_collective_intent_evidence(
    evidence_intent: CommunicationIntent,
) -> None:
    case = compiler_case()
    evidence = EvidenceStore(
        [
            evidence_for(
                case,
                case.explicit_policy.strategy,
                intent=evidence_intent,
            )
        ]
    )

    plan = Compiler(case.registry, evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


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
    assert rank_zero is not rank_one


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


def test_canonical_drift_fails_before_cached_plan_or_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.production_evidence)
    first = compiler.compile(case.intent, case.auto_policy, case.context)
    backend = case.registry.candidates(
        case.intent,
        case.explicit_policy.strategy,
    )[0][1]
    classifiers = dict(compiler_module._CANONICAL_DATACLASS_FIELDS)
    classifiers[CommunicationIntent] = (
        classifiers[CommunicationIntent] - {"rank"}
    )
    monkeypatch.setattr(
        compiler_module,
        "_CANONICAL_DATACLASS_FIELDS",
        classifiers,
    )

    with pytest.raises(CompileError, match="canonical fields"):
        compiler.compile(case.intent, case.auto_policy, case.context)

    assert first.origin is PlanOrigin.AUTO
    assert backend.lower_calls == 1


def test_policy_paths_keep_distinct_cache_entries() -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.evidence)
    native_strategy = exact_native_strategy()
    native = compiler.compile(
        case.intent,
        NativePolicy(),
        case.context,
    )
    explicit = compiler.compile(
        case.intent,
        ExplicitPolicy(native_strategy),
        case.context,
    )
    fallback = compiler.compile(
        case.intent,
        AutoPolicy(),
        case.context,
    )

    assert native.origin is PlanOrigin.NATIVE
    assert explicit.origin is PlanOrigin.EXPLICIT
    assert fallback.origin is PlanOrigin.NATIVE_FALLBACK
    assert len({id(native), id(explicit), id(fallback)}) == 3
    assert compiler.compile(
        case.intent, NativePolicy(), case.context
    ) is native
    assert compiler.compile(
        case.intent, ExplicitPolicy(native_strategy), case.context
    ) is explicit
    assert compiler.compile(
        case.intent, AutoPolicy(), case.context
    ) is fallback

    auto_case = compiler_case()
    auto_compiler = Compiler(
        auto_case.registry,
        auto_case.production_evidence,
    )
    auto = auto_compiler.compile(
        auto_case.intent,
        auto_case.auto_policy,
        auto_case.context,
    )
    assert auto.origin is PlanOrigin.AUTO
    assert auto_compiler.compile(
        auto_case.intent,
        auto_case.auto_policy,
        auto_case.context,
    ) is auto


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
    case = compiler_case()
    record = LegacyEvidenceRecord(
        key=EvidenceKey.from_mapping(
            schema_version=1,
            dimensions={
                "bit_width": "8",
                "bucket_max_bytes": "2048",
                "bucket_min_bytes": "2048",
                "dtype": "float16",
                "error_feedback": "true",
                "group_size": "128",
                "hardware": "a6000",
                "interconnect": "pcie4",
                "logical_bytes": "2048",
                "nodes": "1",
                "output": "full_tensor",
                "overlap": "true",
                "software": "test",
                "strategy": "int8-cag-ring",
                "topology": "ring",
                "wire_bytes": "1056",
                "workload": "communication_bound",
                "world_size": "4",
            },
        ),
        strategy=case.explicit_policy.strategy,
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=LegacyEvidenceMetrics(
            communication_gain_percent=12.0,
            end_to_end_gain_percent=10.0,
            quality_loss_percent=0.5,
            convergence_step_increase_percent=2.0,
            worst_run_gain_percent=-2.0,
            seeds=3,
            cross_workload_reproduced=True,
        ),
    )

    assert compiler_module._record_fingerprint(record) == (
        "535046cd702cc06eb66b24ca1ab83b0f3a9a53407e31be7db9b5909bd435484a"
    )


def test_schema_two_fingerprint_persists_exposed_gain() -> None:
    first = compiler_case().production_evidence.records[0]
    second = replace(
        first,
        metrics=replace(
            first.metrics,
            exposed_communication_gain_percent=2.0,
        ),
    )

    assert compiler_module._record_fingerprint(first) != (
        compiler_module._record_fingerprint(second)
    )


def test_schema_two_evidence_fingerprint_remains_stable() -> None:
    record = compiler_case().production_evidence.records[0]

    assert compiler_module._record_fingerprint(record) == (
        "574825e6d014969aa74acfd8a7549b08ed18737428a368cd9111b627a2729fc1"
    )


@pytest.mark.parametrize(
    "changed_intent",
    [
        replace(compiler_case().intent, reduction=ReductionOp.SUM),
        replace(compiler_case().intent, completion=CompletionMode.SYNC),
        replace(
            compiler_case().intent,
            tensor=TensorSpec(dtype="float16", shape=(32, 32)),
        ),
        replace(
            compiler_case().intent,
            shape_family=ShapeFamily(max_numel=2048, alignment=2),
        ),
    ],
)
def test_schema_two_fingerprint_persists_collective_intent(
    changed_intent: CommunicationIntent,
) -> None:
    case = compiler_case()
    baseline = case.production_evidence.records[0]
    changed = evidence_for(
        case,
        case.explicit_policy.strategy,
        intent=changed_intent,
    )

    assert compiler_module._record_fingerprint(changed) != (
        compiler_module._record_fingerprint(baseline)
    )


def test_schema_two_fingerprint_excludes_local_rank() -> None:
    case = compiler_case()
    baseline = case.production_evidence.records[0]
    changed = evidence_for(
        case,
        case.explicit_policy.strategy,
        intent=replace(case.intent, rank=1),
    )

    assert compiler_module._record_fingerprint(changed) == (
        compiler_module._record_fingerprint(baseline)
    )


def test_schema_one_evidence_never_drives_auto() -> None:
    case = compiler_case()
    current = case.production_evidence.records[0]
    legacy = LegacyEvidenceRecord(
        key=EvidenceKey.from_mapping(
            schema_version=1,
            dimensions=dict(current.key.dimensions),
        ),
        strategy=current.strategy,
        status=EvidenceStatus.PRODUCTION_AUTO,
        metrics=LegacyEvidenceMetrics(
            communication_gain_percent=12.0,
            end_to_end_gain_percent=10.0,
            quality_loss_percent=0.5,
            convergence_step_increase_percent=2.0,
            worst_run_gain_percent=-2.0,
            seeds=3,
            cross_workload_reproduced=True,
        ),
    )

    plan = Compiler(case.registry, EvidenceStore([legacy])).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


def test_evidence_generation_is_deterministic_for_schema_two() -> None:
    case = compiler_case()
    ring = case.production_evidence.records[0]
    tree_strategy = replace(
        case.explicit_policy.strategy,
        topology=TopologyKind.TREE,
    )
    tree = evidence_for(case, tree_strategy)

    assert compiler_module._evidence_generation(
        EvidenceStore([ring, tree])
    ) == compiler_module._evidence_generation(EvidenceStore([tree, ring]))


def test_compiler_falls_back_for_a_forged_frozen_record() -> None:
    case = compiler_case()
    record = case.production_evidence.records[0]
    object.__setattr__(record.metrics, "quality_loss_percent", 5.0)

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


def test_compiler_invalidates_cached_auto_after_record_is_forged() -> None:
    case = compiler_case()
    compiler = Compiler(case.registry, case.production_evidence)
    selected = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )
    record = case.production_evidence.records[0]
    object.__setattr__(record.metrics, "quality_loss_percent", 5.0)

    fallback = compiler.compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert selected.origin is PlanOrigin.AUTO
    assert fallback.origin is PlanOrigin.NATIVE_FALLBACK
    assert fallback is not selected


def test_compiler_falls_back_when_a_frozen_record_key_is_forged() -> None:
    case = compiler_case()
    record = case.production_evidence.records[0]
    object.__setattr__(record, "key", object())

    plan = Compiler(case.registry, case.production_evidence).compile(
        case.intent,
        case.auto_policy,
        case.context,
    )

    assert plan.origin is PlanOrigin.NATIVE_FALLBACK
    assert plan.strategy.compression is CompressionKind.NONE


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
    ("policy_kind", "expected_origin"),
    [
        ("native", PlanOrigin.NATIVE),
        ("explicit", PlanOrigin.EXPLICIT),
        ("auto", PlanOrigin.AUTO),
        ("fallback", PlanOrigin.NATIVE_FALLBACK),
    ],
)
def test_formal_compiler_plans_pass_facade_semantic_validation(
    policy_kind: str,
    expected_origin: PlanOrigin,
) -> None:
    case = compiler_case()
    if policy_kind == "native":
        compiler = Compiler(case.registry, case.evidence)
        policy = NativePolicy()
    elif policy_kind == "explicit":
        compiler = Compiler(case.registry, case.evidence)
        policy = case.explicit_policy
    elif policy_kind == "auto":
        compiler = Compiler(case.registry, case.production_evidence)
        policy = case.auto_policy
    else:
        compiler = Compiler(case.native_registry, case.production_evidence)
        policy = case.auto_policy

    communicator = compile_communicator(
        case.intent,
        policy,
        context=case.context,
        compiler=compiler,
    )

    assert communicator.plan.origin is expected_origin


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
