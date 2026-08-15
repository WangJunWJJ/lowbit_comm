from dataclasses import fields, replace

import pytest

import lowbit_comm.compiler.registry as registry_module
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
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.backends.protocols import BackendCapability
from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
    ExecutionError,
)


def capability(backend_id: str) -> BackendCapability:
    return BackendCapability(
        backend_id=backend_id,
        strategy=strategy(),
        output=OutputSemantics.FULL_TENSOR,
        min_world_size=2,
        max_world_size=8,
        supported_dtypes=frozenset({"float16"}),
        supports_async=True,
    )


class FakeBackend:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self._capability = capability(backend_id)

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (self._capability,)

    def lower(self, intent: object, strategy: object) -> object:
        raise AssertionError("lower is not used by registry unit tests")


class CallableCapabilities:
    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.calls = 0

    def __call__(self) -> tuple[BackendCapability, ...]:
        self.calls += 1
        return (capability(self.backend_id),)


class CallableLower:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, intent: object, strategy: object) -> object:
        del intent, strategy
        self.calls += 1
        raise AssertionError("lower must not run during registration")


class StaticProtocolBackend:
    backend_id = "static"
    capabilities_calls = 0
    lower_calls = 0

    @staticmethod
    def capabilities() -> tuple[BackendCapability, ...]:
        StaticProtocolBackend.capabilities_calls += 1
        return (capability("static"),)

    @staticmethod
    def lower(intent: object, strategy: object) -> object:
        del intent, strategy
        StaticProtocolBackend.lower_calls += 1
        raise AssertionError("lower must not run during registration")


class ClassProtocolBackend:
    backend_id = "class"
    capabilities_calls = 0
    lower_calls = 0

    @classmethod
    def capabilities(cls) -> tuple[BackendCapability, ...]:
        cls.capabilities_calls += 1
        return (capability(cls.backend_id),)

    @classmethod
    def lower(
        cls,
        intent: object,
        strategy: object,
    ) -> object:
        del intent, strategy
        cls.lower_calls += 1
        raise AssertionError("lower must not run during registration")


class InstanceCallableProtocolBackend:
    def __init__(self) -> None:
        self.backend_id = "instance"
        self.capabilities = CallableCapabilities(self.backend_id)
        self.lower = CallableLower()


class SlottedCallableProtocolBackend:
    __slots__ = ("backend_id", "capabilities", "lower")

    def __init__(self) -> None:
        self.backend_id = "slot"
        self.capabilities = CallableCapabilities(self.backend_id)
        self.lower = CallableLower()


class MaliciousStaticMethod(staticmethod):
    func_accesses = 0

    def __getattribute__(self, name: str) -> object:
        if name == "__func__":
            type(self).func_accesses += 1
            raise ValueError("malicious staticmethod accessed")
        return staticmethod.__getattribute__(self, name)


class MaliciousClassMethod(classmethod):
    func_accesses = 0

    def __getattribute__(self, name: str) -> object:
        if name == "__func__":
            type(self).func_accesses += 1
            raise ValueError("malicious classmethod accessed")
        return classmethod.__getattribute__(self, name)


class ExplodingDescriptor:
    accesses = 0

    def __get__(self, instance: object, owner: type[object]) -> object:
        del instance, owner
        type(self).accesses += 1
        raise ValueError("descriptor evaluated")


class CallableExplodingDescriptor(ExplodingDescriptor):
    calls = 0

    def __call__(self, *args: object) -> object:
        del args
        type(self).calls += 1
        raise ValueError("callable descriptor invoked")


def unreachable_capabilities() -> tuple[BackendCapability, ...]:
    raise AssertionError("malicious capabilities invoked")


def unreachable_lower(intent: object, strategy: object) -> object:
    del intent, strategy
    raise AssertionError("malicious lower invoked")


class MissingCapabilitiesBackend:
    backend_id = "cuda"

    def lower(self, intent: object, strategy: object) -> object:
        del intent, strategy
        raise AssertionError("lower must not run during registration")


class MissingLowerBackend:
    backend_id = "cuda"

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (capability("cuda"),)


class NonCallableCapabilitiesBackend(MissingLowerBackend):
    capabilities = object()

    def lower(self, intent: object, strategy: object) -> object:
        del intent, strategy
        raise AssertionError("lower must not run during registration")


class NonCallableLowerBackend(MissingLowerBackend):
    lower = object()


class DynamicCapabilitiesBackend(MissingCapabilitiesBackend):
    def __init__(self) -> None:
        self.getattr_calls = 0

    def __getattr__(self, name: str) -> object:
        self.getattr_calls += 1
        if name == "capabilities":
            return CallableCapabilities(self.backend_id)
        raise AttributeError(name)


class DynamicLowerBackend(MissingLowerBackend):
    def __init__(self) -> None:
        self.getattr_calls = 0

    def __getattr__(self, name: str) -> object:
        self.getattr_calls += 1
        if name == "lower":
            return CallableLower()
        raise AttributeError(name)


class ExplodingCapabilitiesPropertyBackend(MissingCapabilitiesBackend):
    accesses = 0

    @property
    def capabilities(self) -> object:
        type(self).accesses += 1
        raise ValueError("capabilities property evaluated")


class ExplodingLowerPropertyBackend(MissingLowerBackend):
    accesses = 0

    @property
    def lower(self) -> object:
        type(self).accesses += 1
        raise ValueError("lower property evaluated")


class MaliciousStaticCapabilitiesBackend(MissingCapabilitiesBackend):
    capabilities = MaliciousStaticMethod(unreachable_capabilities)


class MaliciousClassLowerBackend(MissingLowerBackend):
    lower = MaliciousClassMethod(unreachable_lower)


class DescriptorCapabilitiesBackend(MissingCapabilitiesBackend):
    capabilities = ExplodingDescriptor()


class DescriptorLowerBackend(MissingLowerBackend):
    lower = ExplodingDescriptor()


class CallableDescriptorCapabilitiesBackend(MissingCapabilitiesBackend):
    capabilities = CallableExplodingDescriptor()


class CallableDescriptorLowerBackend(MissingLowerBackend):
    lower = CallableExplodingDescriptor()


class MissingBackendIdBackend:
    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (capability("cuda"),)

    def lower(self, intent: object, strategy: object) -> object:
        del intent, strategy
        raise AssertionError("lower must not run during registration")


class DynamicBackendIdBackend(MissingLowerBackend):
    def __init__(self) -> None:
        self.getattr_calls = 0

    def __getattr__(self, name: str) -> object:
        self.getattr_calls += 1
        if name == "backend_id":
            return "cuda"
        raise AttributeError(name)


class ExplodingBackendIdPropertyBackend(MissingLowerBackend):
    accesses = 0

    @property
    def backend_id(self) -> object:
        type(self).accesses += 1
        raise ValueError("backend_id property evaluated")


class DescriptorBackendIdBackend(MissingLowerBackend):
    backend_id = ExplodingDescriptor()


class GuardedBackendIdBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__("guarded")
        self.backend_id_accesses = 0

    def __getattribute__(self, name: str) -> object:
        if name == "backend_id":
            accesses = object.__getattribute__(
                self,
                "backend_id_accesses",
            )
            object.__setattr__(
                self,
                "backend_id_accesses",
                accesses + 1,
            )
            raise ValueError("backend_id dynamically accessed")
        return object.__getattribute__(self, name)


class ShadowedInstanceDictBackend:
    backend_id = "shadowed-dict"
    dict_accesses = 0

    @property
    def __dict__(self) -> object:
        type(self).dict_accesses += 1
        raise ValueError("instance dictionary dynamically accessed")

    def capabilities(self) -> tuple[BackendCapability, ...]:
        return (capability(self.backend_id),)

    def lower(self, intent: object, strategy: object) -> object:
        del intent, strategy
        raise AssertionError("lower must not run during registration")


class UninitializedSlottedBackendIdBackend(MissingLowerBackend):
    __slots__ = ("backend_id",)


class ReturningCapabilitiesBackend:
    def __init__(self, returned: object) -> None:
        self.backend_id = "cuda"
        self.returned = returned
        self.capabilities_calls = 0
        self.lower_calls = 0

    def capabilities(self) -> object:
        self.capabilities_calls += 1
        return self.returned

    def lower(self, intent: object, strategy: object) -> object:
        del intent, strategy
        self.lower_calls += 1
        raise AssertionError("lower must not run during registration")


class EqualityTrapBackend(ReturningCapabilitiesBackend):
    def __init__(self, returned: object) -> None:
        super().__init__(returned)
        self.equality_calls = 0

    def __eq__(self, other: object) -> bool:
        del other
        self.equality_calls += 1
        raise AssertionError("backend equality must not be evaluated")


class RaisingCapabilitiesBackend(ReturningCapabilitiesBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__((capability("cuda"),))
        self.error = error

    def capabilities(self) -> object:
        self.capabilities_calls += 1
        raise self.error


class BackendCapabilitySubclass(BackendCapability):
    pass


class StringSubclass(str):
    pass


class InvalidBackendIdBackend(MissingLowerBackend):
    def __init__(self, backend_id: object) -> None:
        self.backend_id = backend_id


def intent(
    *,
    world_size: int = 4,
    completion: CompletionMode = CompletionMode.ASYNC,
) -> CommunicationIntent:
    return CommunicationIntent(
        tensor=TensorSpec(dtype="float16", shape=(16,)),
        shape_family=ShapeFamily(max_numel=16, alignment=1),
        reduction=ReductionOp.SUM,
        output=OutputSemantics.FULL_TENSOR,
        completion=completion,
        world_size=world_size,
        rank=0,
    )


def strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
    )


def strategy_alternatives() -> tuple[tuple[str, StrategySpec], ...]:
    baseline = strategy()
    native = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    return (
        ("compression", native),
        ("collective", native),
        ("topology", replace(baseline, topology=TopologyKind.TREE)),
        ("group_size", replace(baseline, group_size=64)),
        (
            "accumulation_dtype",
            replace(
                baseline,
                accumulation_dtype=AccumulationDType.FP16,
            ),
        ),
        (
            "error_feedback",
            replace(baseline, error_feedback=True),
        ),
        (
            "parameter_error_feedback",
            replace(baseline, parameter_error_feedback=True),
        ),
        ("overlap", replace(baseline, overlap=True)),
        (
            "workspace_budget_bytes",
            replace(baseline, workspace_budget_bytes=1024),
        ),
    )


def assert_registration_rejected_atomically(backend: object) -> None:
    registry = BackendRegistry()

    with pytest.raises(CompileError):
        registry.register(backend)  # type: ignore[arg-type]

    assert registry.generation == 0
    assert registry.capabilities_for_world_size(4) == ()


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda: FakeBackend("cuda"),
        StaticProtocolBackend,
        ClassProtocolBackend,
        InstanceCallableProtocolBackend,
        SlottedCallableProtocolBackend,
    ],
)
def test_registry_accepts_static_safe_backend_protocol_forms(
    backend_factory: object,
) -> None:
    StaticProtocolBackend.capabilities_calls = 0
    StaticProtocolBackend.lower_calls = 0
    ClassProtocolBackend.capabilities_calls = 0
    ClassProtocolBackend.lower_calls = 0
    backend = backend_factory()  # type: ignore[operator]
    registry = BackendRegistry()

    registry.register(backend)

    matches = registry.capabilities_for_world_size(4)
    assert len(matches) == 1
    assert matches[0][1] is backend
    assert registry.generation == 1
    if type(backend) is StaticProtocolBackend:
        assert backend.capabilities_calls == 1
        assert backend.lower_calls == 0
    if type(backend) is ClassProtocolBackend:
        assert backend.capabilities_calls == 1
        assert backend.lower_calls == 0
    if type(backend) in (
        InstanceCallableProtocolBackend,
        SlottedCallableProtocolBackend,
    ):
        assert backend.capabilities.calls == 1
        assert backend.lower.calls == 0


def test_registry_reads_instance_backend_id_without_dynamic_access() -> None:
    backend = GuardedBackendIdBackend()
    registry = BackendRegistry()

    registry.register(backend)

    assert backend.backend_id_accesses == 0
    assert registry.generation == 1


def test_registry_does_not_evaluate_shadowed_instance_dictionary() -> None:
    ShadowedInstanceDictBackend.dict_accesses = 0
    backend = ShadowedInstanceDictBackend()
    registry = BackendRegistry()

    registry.register(backend)

    assert ShadowedInstanceDictBackend.dict_accesses == 0
    assert registry.generation == 1


@pytest.mark.parametrize(
    "backend_factory",
    [
        MissingCapabilitiesBackend,
        MissingLowerBackend,
        NonCallableCapabilitiesBackend,
        NonCallableLowerBackend,
    ],
)
def test_registry_rejects_missing_or_noncallable_protocol_members(
    backend_factory: object,
) -> None:
    assert_registration_rejected_atomically(
        backend_factory(),  # type: ignore[operator]
    )


@pytest.mark.parametrize(
    "backend_factory",
    [DynamicCapabilitiesBackend, DynamicLowerBackend],
)
def test_registry_rejects_dynamic_protocol_members_without_lookup(
    backend_factory: object,
) -> None:
    backend = backend_factory()  # type: ignore[operator]

    assert_registration_rejected_atomically(backend)

    assert backend.getattr_calls == 0


@pytest.mark.parametrize(
    ("backend_factory", "counter_type", "counter_name"),
    [
        (
            ExplodingCapabilitiesPropertyBackend,
            ExplodingCapabilitiesPropertyBackend,
            "accesses",
        ),
        (
            ExplodingLowerPropertyBackend,
            ExplodingLowerPropertyBackend,
            "accesses",
        ),
        (
            MaliciousStaticCapabilitiesBackend,
            MaliciousStaticMethod,
            "func_accesses",
        ),
        (
            MaliciousClassLowerBackend,
            MaliciousClassMethod,
            "func_accesses",
        ),
        (
            DescriptorCapabilitiesBackend,
            ExplodingDescriptor,
            "accesses",
        ),
        (
            DescriptorLowerBackend,
            ExplodingDescriptor,
            "accesses",
        ),
        (
            CallableDescriptorCapabilitiesBackend,
            CallableExplodingDescriptor,
            "calls",
        ),
        (
            CallableDescriptorLowerBackend,
            CallableExplodingDescriptor,
            "calls",
        ),
    ],
)
def test_registry_rejects_descriptors_without_side_effects(
    backend_factory: object,
    counter_type: type[object],
    counter_name: str,
) -> None:
    setattr(counter_type, counter_name, 0)

    assert_registration_rejected_atomically(
        backend_factory(),  # type: ignore[operator]
    )

    assert getattr(counter_type, counter_name) == 0


@pytest.mark.parametrize(
    "backend_factory",
    [
        MissingBackendIdBackend,
        UninitializedSlottedBackendIdBackend,
        lambda: FakeBackend(""),
    ],
)
def test_registry_rejects_missing_or_empty_backend_identifier(
    backend_factory: object,
) -> None:
    assert_registration_rejected_atomically(
        backend_factory(),  # type: ignore[operator]
    )


@pytest.mark.parametrize("backend_id", [1, StringSubclass("cuda")])
def test_registry_requires_exact_string_backend_identifier(
    backend_id: object,
) -> None:
    assert_registration_rejected_atomically(
        InvalidBackendIdBackend(backend_id)
    )


def test_registry_rejects_dynamic_backend_id_without_lookup() -> None:
    backend = DynamicBackendIdBackend()

    assert_registration_rejected_atomically(backend)

    assert backend.getattr_calls == 0


@pytest.mark.parametrize(
    ("backend_factory", "counter_type"),
    [
        (
            ExplodingBackendIdPropertyBackend,
            ExplodingBackendIdPropertyBackend,
        ),
        (DescriptorBackendIdBackend, ExplodingDescriptor),
    ],
)
def test_registry_rejects_backend_id_descriptors_without_side_effects(
    backend_factory: object,
    counter_type: type[object],
) -> None:
    counter_type.accesses = 0  # type: ignore[attr-defined]

    assert_registration_rejected_atomically(
        backend_factory(),  # type: ignore[operator]
    )

    assert counter_type.accesses == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "returned",
    [
        (),
        [],
        (object(),),
        (capability("cuda"), object()),
        (
            BackendCapabilitySubclass(
                backend_id="cuda",
                strategy=strategy(),
                output=OutputSemantics.FULL_TENSOR,
                min_world_size=2,
                max_world_size=8,
                supported_dtypes=frozenset({"float16"}),
                supports_async=True,
            ),
        ),
    ],
)
def test_registry_rejects_invalid_capability_collections_atomically(
    returned: object,
) -> None:
    backend = ReturningCapabilitiesBackend(returned)

    assert_registration_rejected_atomically(backend)

    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0


def test_registry_revalidates_exact_capability_instances() -> None:
    forged = capability("cuda")
    object.__setattr__(forged, "supported_dtypes", {"float16"})
    backend = ReturningCapabilitiesBackend((forged,))

    assert_registration_rejected_atomically(backend)

    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0


def test_failed_registration_preserves_existing_entries_and_generation(
) -> None:
    registered = FakeBackend("cpu")
    registry = BackendRegistry([registered])
    before = registry.capabilities_for_world_size(4)
    backend = ReturningCapabilitiesBackend(
        (capability("cuda"), object())
    )

    with pytest.raises(CompileError):
        registry.register(backend)  # type: ignore[arg-type]

    assert registry.generation == 1
    assert registry.capabilities_for_world_size(4) == before
    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        ValueError("value failure"),
        CapabilityError("capability failure"),
        ExecutionError("execution failure"),
    ],
)
def test_registry_normalizes_capabilities_failure(
    error: Exception,
) -> None:
    backend = RaisingCapabilitiesBackend(error)

    registry = BackendRegistry()

    with pytest.raises(
        CompileError,
        match="capabilities.*failed during registration",
    ):
        registry.register(backend)  # type: ignore[arg-type]

    assert registry.generation == 0
    assert registry.capabilities_for_world_size(4) == ()
    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0


def test_registry_revalidates_nested_strategy_in_capability() -> None:
    forged = capability("cuda")
    object.__setattr__(forged.strategy, "group_size", -1)
    backend = ReturningCapabilitiesBackend((forged,))

    assert_registration_rejected_atomically(backend)

    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0


def test_registry_validates_lower_before_invoking_capabilities() -> None:
    backend = ReturningCapabilitiesBackend((capability("cuda"),))
    backend.lower = object()  # type: ignore[method-assign]

    assert_registration_rejected_atomically(backend)

    assert backend.capabilities_calls == 0
    assert backend.lower_calls == 0


def test_registry_calls_capabilities_once_and_never_calls_lower() -> None:
    backend = ReturningCapabilitiesBackend((capability("cuda"),))
    registry = BackendRegistry()

    registry.register(backend)

    assert backend.capabilities_calls == 1
    assert backend.lower_calls == 0
    assert registry.generation == 1


@pytest.mark.parametrize(
    "alternative",
    [
        replace(
            capability("cuda"),
            supported_dtypes=frozenset({"float32"}),
        ),
        replace(
            capability("cuda"),
            min_world_size=9,
            max_world_size=16,
        ),
        replace(
            capability("cuda"),
            strategy=replace(strategy(), topology=TopologyKind.TREE),
        ),
    ],
    ids=["dtype", "world-size-range", "strategy"],
)
def test_registry_rejects_non_owner_before_capability_access(
    alternative: BackendCapability,
) -> None:
    owner = EqualityTrapBackend((capability("cuda"),))
    contender = EqualityTrapBackend((alternative,))
    registry = BackendRegistry([owner])
    before = registry.capabilities_for_world_size(4)

    with pytest.raises(
        CapabilityError,
        match="identifier.*owned by another backend object",
    ):
        registry.register(contender)

    assert contender.capabilities_calls == 0
    assert contender.lower_calls == 0
    assert owner.equality_calls == contender.equality_calls == 0
    assert registry.generation == 1
    assert registry.capabilities_for_world_size(4) == before


@pytest.mark.parametrize(
    ("backend_factory", "counter_type"),
    [
        (
            ExplodingCapabilitiesPropertyBackend,
            ExplodingCapabilitiesPropertyBackend,
        ),
        (ExplodingLowerPropertyBackend, ExplodingLowerPropertyBackend),
        (DescriptorCapabilitiesBackend, ExplodingDescriptor),
        (DescriptorLowerBackend, ExplodingDescriptor),
    ],
)
def test_registry_owner_collision_has_no_protocol_descriptor_effects(
    backend_factory: object,
    counter_type: type[object],
) -> None:
    registry = BackendRegistry([FakeBackend("cuda")])
    counter_type.accesses = 0  # type: ignore[attr-defined]

    with pytest.raises(CapabilityError, match="identifier.*owned"):
        registry.register(backend_factory())  # type: ignore[operator]

    assert counter_type.accesses == 0  # type: ignore[attr-defined]
    assert registry.generation == 1


def test_same_backend_object_can_register_non_overlapping_batches() -> None:
    baseline = capability("cuda")
    alternative = replace(
        baseline,
        strategy=replace(baseline.strategy, topology=TopologyKind.TREE),
    )
    backend = ReturningCapabilitiesBackend((baseline,))
    registry = BackendRegistry()

    registry.register(backend)
    backend.returned = (alternative,)
    registry.register(backend)

    matches = registry.capabilities_for_world_size(4)
    assert tuple(match[0] for match in matches) == (
        baseline,
        alternative,
    )
    assert all(match[1] is backend for match in matches)
    assert backend.capabilities_calls == 2
    assert backend.lower_calls == 0
    assert registry.generation == 2

    backend.returned = (baseline,)
    with pytest.raises(
        CapabilityError,
        match="Duplicate backend capability key",
    ):
        registry.register(backend)

    assert backend.capabilities_calls == 3
    assert registry.generation == 2


@pytest.mark.parametrize(
    "returned",
    [(), (object(),), (capability("cuda"), object())],
)
def test_failed_first_batch_does_not_claim_backend_identity(
    returned: object,
) -> None:
    failed = ReturningCapabilitiesBackend(returned)
    owner = FakeBackend("cuda")
    registry = BackendRegistry()

    with pytest.raises(CompileError):
        registry.register(failed)
    registry.register(owner)

    assert registry.resolve_exact(capability("cuda"))[1] is owner
    assert registry.generation == 1


def test_inconsistent_committed_backend_ownership_fails_closed() -> None:
    baseline = capability("cuda")
    alternative = replace(
        baseline,
        strategy=replace(baseline.strategy, topology=TopologyKind.TREE),
    )
    backend = ReturningCapabilitiesBackend((baseline, alternative))
    registry = BackendRegistry([backend])
    keys = tuple(registry._entries)
    forged_owner = FakeBackend("cuda")
    registry._entries[keys[1]] = registry._entries[keys[1]]._replace(
        backend=forged_owner,
        lower=forged_owner.lower,
    )
    before = dict(registry._entries)

    backend.returned = (
        replace(
            baseline,
            strategy=replace(baseline.strategy, group_size=64),
        ),
    )
    with pytest.raises(
        CapabilityError,
        match="identity ownership is inconsistent",
    ):
        registry.register(backend)

    assert backend.capabilities_calls == 1
    assert registry._entries == before
    assert registry.generation == 1


def test_reference_oracle_cannot_be_registered_as_a_backend() -> None:
    registry = BackendRegistry()

    with pytest.raises(CompileError):
        registry.register(ReferenceBackend())  # type: ignore[arg-type]

    assert registry.generation == 0
    assert registry.capabilities_for_world_size(4) == ()


class StrategySpecSubclass(StrategySpec):
    pass


def test_strategy_cases_cover_every_strategy_field() -> None:
    assert {name for name, _ in strategy_alternatives()} == {
        field.name for field in fields(StrategySpec)
    }


def test_registry_rejects_duplicate_capability_key() -> None:
    registry = BackendRegistry()

    registry.register(FakeBackend("cuda"))
    before = registry.capabilities_for_world_size(4)

    with pytest.raises(CapabilityError):
        registry.register(FakeBackend("cuda"))

    assert registry.generation == 1
    assert registry.capabilities_for_world_size(4) == before


def test_diagnostic_world_size_lookup_excludes_out_of_range_capabilities(
) -> None:
    registry = BackendRegistry([FakeBackend("cuda")])

    assert registry.capabilities_for_world_size(world_size=16) == ()


def test_capability_matches_an_exact_intent_and_strategy() -> None:
    assert capability("cuda").supports(intent(), strategy())


def test_registry_returns_capability_and_backend_in_key_order() -> None:
    cuda = FakeBackend("cuda")
    cpu = FakeBackend("cpu")
    registry = BackendRegistry([cuda, cpu])

    matches = registry.candidates(intent(), strategy())

    assert matches == ((capability("cpu"), cpu), (capability("cuda"), cuda))
    assert registry.resolve_exact(capability("cuda")) == (
        capability("cuda"),
        cuda,
    )


def test_registration_owns_a_fully_independent_capability_snapshot() -> None:
    advertised = capability("cuda")
    backend = ReturningCapabilitiesBackend((advertised,))
    registry = BackendRegistry([backend])
    key, entry = next(iter(registry._entries.items()))

    assert entry.capability == advertised
    assert entry.capability is not advertised
    assert entry.capability.strategy is not advertised.strategy
    assert entry.capability.supported_dtypes is not advertised.supported_dtypes
    assert key == registry_module._capability_key(entry.capability)


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("backend_id", "forged"),
        ("output", OutputSemantics.REDUCED_SHARD),
        ("min_world_size", 5),
        ("max_world_size", None),
        ("supported_dtypes", frozenset({"float32"})),
        ("supports_async", False),
    ],
)
def test_mutating_backend_capability_after_registration_is_isolated(
    field: str,
    mutated: object,
) -> None:
    expected = capability("cuda")
    advertised = capability("cuda")
    backend = ReturningCapabilitiesBackend((advertised,))
    registry = BackendRegistry([backend])

    object.__setattr__(advertised, field, mutated)

    assert registry.resolve_exact(expected) == (expected, backend)
    assert registry.capabilities_for_world_size(4) == ((expected, backend),)
    assert registry.generation == 1


@pytest.mark.parametrize(("field", "alternative"), strategy_alternatives())
def test_mutating_backend_strategy_after_registration_is_isolated(
    field: str,
    alternative: StrategySpec,
) -> None:
    expected = capability("cuda")
    advertised = capability("cuda")
    registry = BackendRegistry(
        [ReturningCapabilitiesBackend((advertised,))]
    )

    object.__setattr__(
        advertised.strategy,
        field,
        getattr(alternative, field),
    )

    assert registry.resolve_exact(expected)[0] == expected
    assert registry.candidates(intent(), strategy())[0][0] == expected
    assert registry.generation == 1


def test_every_registry_capability_result_is_a_fresh_snapshot() -> None:
    expected = capability("cuda")
    backend = FakeBackend("cuda")
    registry = BackendRegistry([backend])
    internal = next(iter(registry._entries.values())).capability

    candidate = registry.candidates(intent(), strategy())[0][0]
    resolved = registry.resolve_exact(expected)[0]
    diagnostic = registry.capabilities_for_world_size(4)[0][0]
    lowering = registry._resolve_lowering(expected)[0]
    exposed = (candidate, resolved, diagnostic, lowering)

    assert all(item == expected for item in exposed)
    assert all(item is not internal for item in exposed)
    assert len({id(item) for item in exposed}) == len(exposed)
    assert len({id(item.strategy) for item in exposed}) == len(exposed)
    assert len({id(item.supported_dtypes) for item in exposed}) == len(
        exposed
    )


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("backend_id", "forged"),
        ("output", OutputSemantics.REDUCED_SHARD),
        ("min_world_size", 5),
        ("max_world_size", None),
        ("supported_dtypes", frozenset({"float32"})),
        ("supports_async", False),
    ],
)
def test_mutating_public_capability_does_not_change_registry_state(
    field: str,
    mutated: object,
) -> None:
    expected = capability("cuda")
    backend = FakeBackend("cuda")
    registry = BackendRegistry([backend])
    exposed = registry.candidates(intent(), strategy())[0][0]

    object.__setattr__(exposed, field, mutated)

    assert registry.resolve_exact(expected) == (expected, backend)
    assert registry.candidates(intent(), strategy()) == ((expected, backend),)
    assert registry.generation == 1


@pytest.mark.parametrize(("field", "alternative"), strategy_alternatives())
def test_mutating_public_strategy_does_not_change_registry_state(
    field: str,
    alternative: StrategySpec,
) -> None:
    expected = capability("cuda")
    backend = FakeBackend("cuda")
    registry = BackendRegistry([backend])
    exposed = registry.resolve_exact(expected)[0]

    object.__setattr__(
        exposed.strategy,
        field,
        getattr(alternative, field),
    )

    assert registry.resolve_exact(expected) == (expected, backend)
    assert registry.candidates(intent(), strategy()) == ((expected, backend),)
    assert registry.generation == 1


def test_registry_fails_closed_when_internal_snapshot_key_drifts() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])
    entry = next(iter(registry._entries.values()))
    object.__setattr__(entry.capability, "backend_id", "forged")

    with pytest.raises(CompileError, match="internal capability"):
        registry.capabilities_for_world_size(4)

    assert registry.generation == 1


def test_registry_internal_entry_prevents_lower_anchor_mutation() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])
    entry = next(iter(registry._entries.values()))
    original_lower = entry.lower
    trap_calls = 0

    assert type(entry).__bases__ == (tuple,)
    assert type(entry).__slots__ == ()
    assert not hasattr(entry, "__dict__")

    def trap_lower(intent: object, strategy: object) -> object:
        nonlocal trap_calls
        del intent, strategy
        trap_calls += 1
        return object()

    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(entry, "lower", trap_lower)

    assert registry._resolve_lowering(capability("cuda"))[1] is original_lower
    assert trap_calls == 0


def test_registry_rejects_replaced_exact_entry_with_forged_lower() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])
    key, entry = next(iter(registry._entries.items()))
    trap_calls = 0

    def trap_lower(intent: object, strategy: object) -> object:
        nonlocal trap_calls
        del intent, strategy
        trap_calls += 1
        return object()

    registry._entries[key] = type(entry)(
        entry.capability,
        entry.backend,
        trap_lower,
    )

    with pytest.raises(CompileError, match="internal capability"):
        registry._resolve_lowering(capability("cuda"))

    assert registry.generation == 1
    assert trap_calls == 0


def test_registry_rejects_raw_tuple_entry_replacement() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])
    key, entry = next(iter(registry._entries.items()))
    registry._entries[key] = (  # type: ignore[assignment]
        entry.capability,
        entry.backend,
        entry.lower,
    )

    with pytest.raises(CompileError, match="internal capability"):
        registry._resolve_lowering(capability("cuda"))

    assert registry.generation == 1


def test_registry_resolves_saved_lowering_without_changing_public_match(
) -> None:
    backend = FakeBackend("cuda")
    registry = BackendRegistry([backend])
    selected = registry.candidates(intent(), strategy())[0]

    lowering = registry._resolve_lowering(selected[0])

    assert len(selected) == 2
    assert selected == (capability("cuda"), backend)
    assert lowering[0] == selected[0]
    assert lowering[0] is not selected[0]
    assert callable(lowering[1])


def test_registry_shares_one_bound_lower_across_backend_capabilities() -> None:
    baseline = capability("cuda")
    alternative = replace(baseline, max_world_size=None)

    class MultiCapabilityBackend:
        backend_id = "cuda"

        def __init__(self) -> None:
            self.capabilities_calls = 0

        def capabilities(self) -> tuple[BackendCapability, ...]:
            self.capabilities_calls += 1
            return baseline, alternative

        def lower(self, request: object, spec: object) -> object:
            del request, spec
            raise AssertionError("lower is not used by registry unit tests")

    backend = MultiCapabilityBackend()
    registry = BackendRegistry([backend])
    matches = registry.capabilities_for_world_size(4)

    lowerings = tuple(
        registry._resolve_lowering(candidate)
        for candidate, _ in matches
    )

    assert backend.capabilities_calls == 1
    assert registry.generation == 1
    assert len(lowerings) == 2
    assert lowerings[0][1] is lowerings[1][1]


def test_registry_rejects_missing_or_non_exact_lowering_lookup() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])

    with pytest.raises(CapabilityError):
        registry._resolve_lowering(capability("cpu"))
    with pytest.raises(CompileError):
        registry._resolve_lowering(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_id", StringSubclass("cuda")),
        ("supported_dtypes", {"float16"}),
    ],
)
def test_registry_rejects_same_key_forged_lowering_lookup(
    field: str,
    value: object,
) -> None:
    registered = capability("cuda")
    forged = replace(registered)
    object.__setattr__(forged, field, value)
    backend = FakeBackend("cuda")
    registry = BackendRegistry([backend])

    with pytest.raises(CompileError):
        registry._resolve_lowering(forged)

    assert registry.generation == 1
    assert registry.capabilities_for_world_size(4) == (
        (capability("cuda"), backend),
    )


def test_registry_sorts_bounded_and_unbounded_capability_keys() -> None:
    bounded = capability("cuda")
    unbounded = BackendCapability(
        backend_id="cuda",
        strategy=bounded.strategy,
        output=bounded.output,
        min_world_size=bounded.min_world_size,
        max_world_size=None,
        supported_dtypes=bounded.supported_dtypes,
        supports_async=bounded.supports_async,
    )

    class MultiCapabilityBackend:
        backend_id = "cuda"

        def capabilities(self) -> tuple[BackendCapability, ...]:
            return (unbounded, bounded)

        def lower(self, request: object, spec: object) -> object:
            raise AssertionError("lower is not used by registry unit tests")

    matches = BackendRegistry([MultiCapabilityBackend()]).candidates(
        intent(), strategy()
    )

    assert tuple(match[0] for match in matches) == (bounded, unbounded)


def test_registry_order_is_independent_of_strategy_registration_order(
) -> None:
    without_workspace = capability("cuda")
    with_workspace = replace(
        without_workspace,
        strategy=replace(
            without_workspace.strategy,
            workspace_budget_bytes=0,
        ),
    )

    class MultiCapabilityBackend:
        backend_id = "cuda"

        def __init__(
            self,
            capabilities: tuple[BackendCapability, ...],
        ) -> None:
            self._capabilities = capabilities

        def capabilities(self) -> tuple[BackendCapability, ...]:
            return self._capabilities

        def lower(self, request: object, spec: object) -> object:
            raise AssertionError("lower is not used by registry unit tests")

    forward = BackendRegistry(
        [MultiCapabilityBackend((without_workspace, with_workspace))]
    ).capabilities_for_world_size(4)
    reverse = BackendRegistry(
        [MultiCapabilityBackend((with_workspace, without_workspace))]
    ).capabilities_for_world_size(4)

    assert tuple(match[0] for match in forward) == tuple(
        match[0] for match in reverse
    )
    assert len(forward) == 2


def test_registry_rejects_unfiltered_candidate_lookup() -> None:
    registry = BackendRegistry([FakeBackend("cuda")])

    with pytest.raises(TypeError):
        registry.candidates()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "candidate_intent,candidate_strategy",
    [(object(), strategy()), (intent(), object())],
)
def test_registry_rejects_non_exact_compile_contracts(
    candidate_intent: object,
    candidate_strategy: object,
) -> None:
    registry = BackendRegistry([FakeBackend("cuda")])

    with pytest.raises(CompileError):
        registry.candidates(
            candidate_intent,  # type: ignore[arg-type]
            candidate_strategy,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "candidate_intent,candidate_strategy",
    [(object(), strategy()), (intent(), object())],
)
def test_capability_does_not_support_non_exact_compile_contracts(
    candidate_intent: object,
    candidate_strategy: object,
) -> None:
    assert not capability("cuda").supports(
        candidate_intent,  # type: ignore[arg-type]
        candidate_strategy,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("output", OutputSemantics.REDUCED_SHARD),
        ("supported_dtypes", frozenset({"float32"})),
        ("min_world_size", 5),
        ("max_world_size", 3),
        ("supports_async", False),
    ],
)
def test_capability_rejects_non_matching_compile_request(
    field: str,
    value: object,
) -> None:
    candidate = replace(capability("cuda"), **{field: value})

    assert not candidate.supports(intent(), strategy())


@pytest.mark.parametrize(("field", "candidate"), strategy_alternatives())
def test_capability_rejects_every_strategy_field_difference(
    field: str,
    candidate: StrategySpec,
) -> None:
    assert getattr(candidate, field) != getattr(strategy(), field)
    assert not capability("cuda").supports(intent(), candidate)
    assert BackendRegistry([FakeBackend("cuda")]).candidates(
        intent(), candidate
    ) == ()


@pytest.mark.parametrize(("field", "candidate"), strategy_alternatives())
def test_registry_key_includes_every_strategy_field(
    field: str,
    candidate: StrategySpec,
) -> None:
    baseline = capability("cuda")
    alternative = replace(baseline, strategy=candidate)

    class MultiCapabilityBackend:
        backend_id = "cuda"

        def capabilities(self) -> tuple[BackendCapability, ...]:
            return (baseline, alternative)

        def lower(self, request: object, spec: object) -> object:
            raise AssertionError("lower is not used by registry unit tests")

    registry = BackendRegistry([MultiCapabilityBackend()])

    assert getattr(candidate, field) != getattr(baseline.strategy, field)
    resolved_baseline = registry.resolve_exact(baseline)[0]
    resolved_alternative = registry.resolve_exact(alternative)[0]
    assert resolved_baseline == baseline
    assert resolved_baseline is not baseline
    assert resolved_alternative == alternative
    assert resolved_alternative is not alternative


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend_id", 1),
        ("strategy", object()),
        ("min_world_size", True),
        ("max_world_size", False),
        ("supported_dtypes", {"float16"}),
        ("supports_async", 1),
    ],
)
def test_capability_rejects_non_exact_deterministic_fields(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "backend_id": "cuda",
        "strategy": strategy(),
        "output": OutputSemantics.FULL_TENSOR,
        "min_world_size": 2,
        "max_world_size": 8,
        "supported_dtypes": frozenset({"float16"}),
        "supports_async": True,
    }
    values[field] = value

    with pytest.raises(CompileError):
        BackendCapability(**values)  # type: ignore[arg-type]


def test_capability_rejects_strategy_subclasses() -> None:
    subclass = StrategySpecSubclass(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
    )

    with pytest.raises(CompileError):
        replace(capability("cuda"), strategy=subclass)
    assert not capability("cuda").supports(intent(), subclass)


def test_capability_rejects_inverted_world_size_bounds() -> None:
    with pytest.raises(CompileError):
        BackendCapability(
            backend_id="cuda",
            strategy=strategy(),
            output=OutputSemantics.FULL_TENSOR,
            min_world_size=8,
            max_world_size=2,
            supported_dtypes=frozenset({"float16"}),
            supports_async=True,
        )
