from dataclasses import fields, replace

import pytest

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
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import CapabilityError, CompileError


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


class StrategySpecSubclass(StrategySpec):
    pass


def test_strategy_cases_cover_every_strategy_field() -> None:
    assert {name for name, _ in strategy_alternatives()} == {
        field.name for field in fields(StrategySpec)
    }


def test_registry_rejects_duplicate_capability_key() -> None:
    registry = BackendRegistry()

    registry.register(FakeBackend("cuda"))

    with pytest.raises(CapabilityError):
        registry.register(FakeBackend("cuda"))


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
    assert registry.resolve_exact(baseline)[0] is baseline
    assert registry.resolve_exact(alternative)[0] is alternative


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
