import pytest

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
from lowbit_comm.core.errors import CompileError


class UnhashableInt(int):
    __hash__ = None


class UnhashableFrozenSet(frozenset[object]):
    __hash__ = None


class StrategySpecSubclass(StrategySpec):
    __hash__ = None


class AutoConstraintsSubclass(AutoConstraints):
    __hash__ = None


def _strategy_with_group_size(value: UnhashableInt) -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=value,  # type: ignore[arg-type]
    )


def _strategy_with_workspace_budget(value: UnhashableInt) -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=128,
        workspace_budget_bytes=value,  # type: ignore[arg-type]
    )


def _constraints_with_workspace_budget(
    value: UnhashableInt,
) -> AutoConstraints:
    return AutoConstraints(max_workspace_bytes=value)  # type: ignore[arg-type]


def test_explicit_policy_preserves_exact_user_strategy() -> None:
    spec = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        error_feedback=True,
        overlap=True,
    )

    assert ExplicitPolicy(spec).strategy is spec


def test_auto_constraints_are_immutable_sets() -> None:
    constraints = AutoConstraints(
        allowed_compressions=frozenset({CompressionKind.INT8}),
        denied_topologies=frozenset({TopologyKind.TREE}),
        max_workspace_bytes=256 * 1024 * 1024,
    )

    assert AutoPolicy(constraints).constraints.max_workspace_bytes == 268435456
    assert hash(constraints)


def test_int8_requires_positive_group_size() -> None:
    with pytest.raises(CompileError):
        StrategySpec(
            compression=CompressionKind.INT8,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=0,
        )


def test_none_compression_rejects_a_group_size() -> None:
    with pytest.raises(CompileError):
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=128,
        )


def test_strategy_spec_is_immutable_and_hashable() -> None:
    spec = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
        accumulation_dtype=AccumulationDType.FP32,
        error_feedback=True,
        parameter_error_feedback=False,
        overlap=True,
    )

    assert hash(spec)
    with pytest.raises(AttributeError):
        spec.overlap = True


def test_policies_are_immutable_and_hashable() -> None:
    native = NativePolicy()
    auto = AutoPolicy()
    explicit = ExplicitPolicy(
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
        )
    )

    assert hash(native)
    assert hash(auto)
    assert hash(explicit)
    with pytest.raises(AttributeError):
        auto.constraints = AutoConstraints()


@pytest.mark.parametrize(
    "compression,collective",
    [
        (CompressionKind.NONE, CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE),
        (CompressionKind.INT8, CollectiveKind.NATIVE),
    ],
)
def test_strategy_spec_rejects_intrinsically_contradictory_combinations(
    compression: CompressionKind,
    collective: CollectiveKind,
) -> None:
    with pytest.raises(CompileError):
        StrategySpec(
            compression=compression,
            collective=collective,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=128,
        )


@pytest.mark.parametrize("budget", [-1, -1024])
def test_strategy_spec_rejects_negative_workspace_budget(budget: int) -> None:
    with pytest.raises(CompileError):
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
            workspace_budget_bytes=budget,
        )


@pytest.mark.parametrize("constraints", [set(), {CompressionKind.INT8}])
def test_auto_constraints_rejects_mutable_compression_inputs(
    constraints: set[CompressionKind],
) -> None:
    with pytest.raises(CompileError):
        AutoConstraints(allowed_compressions=constraints)


def test_auto_constraints_rejects_mutable_topology_inputs() -> None:
    with pytest.raises(CompileError):
        AutoConstraints(denied_topologies={TopologyKind.RING})


@pytest.mark.parametrize(
    "denied_constraint",
    [
        "denied_compressions",
        "denied_collectives",
        "denied_topologies",
    ],
)
def test_auto_constraints_rejects_missing_denied_sets(
    denied_constraint: str,
) -> None:
    with pytest.raises(CompileError):
        AutoConstraints(**{denied_constraint: None})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[], 1])
@pytest.mark.parametrize(
    "flag_name",
    ["error_feedback", "parameter_error_feedback", "overlap"],
)
def test_strategy_spec_rejects_non_boolean_flags(
    flag_name: str,
    value: object,
) -> None:
    with pytest.raises(CompileError):
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
            **{flag_name: value},
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "constructor",
    [
        _strategy_with_group_size,
        _strategy_with_workspace_budget,
        _constraints_with_workspace_budget,
    ],
)
def test_policy_contracts_reject_unhashable_integer_subclasses(
    constructor: object,
) -> None:
    with pytest.raises(CompileError):
        constructor(UnhashableInt(128))  # type: ignore[operator]


@pytest.mark.parametrize(
    "field_name,values",
    [
        (
            "allowed_compressions",
            UnhashableFrozenSet({CompressionKind.INT8}),
        ),
        (
            "denied_topologies",
            UnhashableFrozenSet({TopologyKind.TREE}),
        ),
    ],
)
def test_auto_constraints_reject_unhashable_frozenset_subclasses(
    field_name: str,
    values: UnhashableFrozenSet,
) -> None:
    with pytest.raises(CompileError):
        AutoConstraints(**{field_name: values})  # type: ignore[arg-type]


def test_policies_reject_contract_subclasses() -> None:
    strategy = StrategySpecSubclass(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )
    constraints = AutoConstraintsSubclass()

    with pytest.raises(CompileError):
        ExplicitPolicy(strategy)
    with pytest.raises(CompileError):
        AutoPolicy(constraints)
