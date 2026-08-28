"""Contract tests for deterministic all-rank reference execution."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import lowbit_comm.backends.reference.backend as reference_module
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
from lowbit_comm.api.result import (
    FullTensorResult,
    ReducedShardResult,
)
from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.backends.reference.backend import ReferenceGroupPlan
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import (
    CompileError,
    ExecutionError,
)
from lowbit_comm.runtime.work import FailedWork


ReferenceResult = (
    FullTensorResult[tuple[float, ...]] | ReducedShardResult[tuple[float, ...]]
)


class FloatSubclass(float):
    """A float subclass that must not cross the exact scalar boundary."""


class TupleSubclass(tuple[object, ...]):
    """A tuple subclass that must not cross a container boundary."""


class IntentSubclass(CommunicationIntent):
    """An intent subclass that must not cross the exact contract boundary."""


class StringSubclass(str):
    """A string subclass that must not satisfy exact tensor contracts."""


class StrategySubclass(StrategySpec):
    """A strategy subclass that must not cross exact graph boundaries."""


class PlanSubclass(ReferenceGroupPlan):
    """A plan subclass that must not cross exact execution boundaries."""


def _forge_intent(intent: CommunicationIntent, scenario: str) -> None:
    if scenario == "tensor-shape":
        object.__setattr__(intent.tensor, "shape", ())
    elif scenario == "tensor-dtype":
        object.__setattr__(intent.tensor, "dtype", ["float16"])
    elif scenario == "tensor-dtype-subclass":
        object.__setattr__(intent.tensor, "dtype", StringSubclass("float16"))
    elif scenario == "shape-family":
        object.__setattr__(intent.shape_family, "max_numel", -1)
    elif scenario == "world-size":
        object.__setattr__(intent, "world_size", 0)
    elif scenario == "rank":
        object.__setattr__(intent, "rank", intent.world_size)
    elif scenario == "output":
        object.__setattr__(intent, "output", object())
    else:
        object.__setattr__(intent, "reduction", object())


def make_intent(
    *,
    output: OutputSemantics,
    reduction: ReductionOp = ReductionOp.SUM,
    tensor_shape: tuple[int, ...] = (2,),
    world_size: int = 4,
    completion: CompletionMode = CompletionMode.SYNC,
    dtype: str = "float16",
) -> CommunicationIntent:
    """Build one exact communication intent for a reference test."""
    numel = 1
    for size in tensor_shape:
        numel *= size
    return CommunicationIntent(
        tensor=TensorSpec(dtype=dtype, shape=tensor_shape),
        shape_family=ShapeFamily(max_numel=numel, alignment=1),
        reduction=reduction,
        output=output,
        completion=completion,
        world_size=world_size,
        rank=0,
    )


def native_strategy() -> StrategySpec:
    """Return the only strategy implemented by the reference backend."""
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )


def test_reference_fulltensor_mean_is_identical_on_every_rank() -> None:
    intent = make_intent(
        output=OutputSemantics.FULL_TENSOR,
        reduction=ReductionOp.MEAN,
    )

    results = ReferenceBackend().execute_group(
        intent,
        rank_values=(
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
            (7.0, 8.0),
        ),
    )

    assert all(type(result) is FullTensorResult for result in results)
    full_results = cast(
        tuple[FullTensorResult[tuple[float, ...]], ...],
        results,
    )
    assert [result.value for result in full_results] == [(4.0, 5.0)] * 4
    assert len({id(result) for result in full_results}) == 4
    with pytest.raises(FrozenInstanceError):
        full_results[0].value = ()


def test_reference_fulltensor_sum_uses_logical_multidimensional_numel() -> None:
    intent = make_intent(
        output=OutputSemantics.FULL_TENSOR,
        tensor_shape=(2, 2),
        world_size=2,
    )

    results = ReferenceBackend().execute_group(
        intent,
        rank_values=(
            (1.0, 2.0, 3.0, 4.0),
            (10.0, 20.0, 30.0, 40.0),
        ),
    )

    assert tuple(result.value for result in results) == (
        (11.0, 22.0, 33.0, 44.0),
        (11.0, 22.0, 33.0, 44.0),
    )


def test_reference_reduced_shard_has_complete_uneven_ownership() -> None:
    intent = make_intent(
        output=OutputSemantics.REDUCED_SHARD,
        tensor_shape=(10,),
    )
    rank_values = tuple(
        tuple(float(rank + index) for index in range(10)) for rank in range(4)
    )

    results = ReferenceBackend().execute_group(intent, rank_values)

    assert all(type(result) is ReducedShardResult for result in results)
    shard_results = cast(
        tuple[ReducedShardResult[tuple[float, ...]], ...],
        results,
    )
    owned = [
        index
        for result in shard_results
        for index in range(result.metadata.offset, result.metadata.stop)
    ]
    assert owned == list(range(10))
    assert [result.metadata.offset for result in shard_results] == [
        0,
        3,
        6,
        9,
    ]
    assert [result.metadata.valid_length for result in shard_results] == [
        3,
        3,
        3,
        1,
    ]
    assert [result.metadata.padded_length for result in shard_results] == [
        3,
        3,
        3,
        3,
    ]
    assert shard_results[-1].value == (42.0, 0.0, 0.0)
    assert all(
        len(result.value) == result.metadata.padded_length for result in shard_results
    )
    assert all(
        result.metadata.global_shape == (10,) and result.metadata.owner_rank == rank
        for rank, result in enumerate(shard_results)
    )


def test_reference_reduced_shard_mean_is_applied_before_partitioning() -> None:
    intent = make_intent(
        output=OutputSemantics.REDUCED_SHARD,
        reduction=ReductionOp.MEAN,
        tensor_shape=(5,),
        world_size=2,
    )

    results = ReferenceBackend().execute_group(
        intent,
        rank_values=(
            (1.0, 3.0, 5.0, 7.0, 9.0),
            (3.0, 5.0, 7.0, 9.0, 11.0),
        ),
    )

    assert tuple(result.value for result in results) == (
        (2.0, 4.0, 6.0),
        (8.0, 10.0, 0.0),
    )


def test_reference_reduced_shard_supports_more_ranks_than_elements() -> None:
    intent = make_intent(
        output=OutputSemantics.REDUCED_SHARD,
        tensor_shape=(2,),
    )

    results = ReferenceBackend().execute_group(
        intent,
        rank_values=((1.0, 2.0),) * 4,
    )

    assert tuple(result.value for result in results) == (
        (4.0,),
        (8.0,),
        (0.0,),
        (0.0,),
    )
    assert tuple(
        (
            result.metadata.offset,
            result.metadata.stop,
            result.metadata.valid_length,
            result.metadata.padded_length,
        )
        for result in results
    ) == (
        (0, 1, 1, 1),
        (1, 2, 1, 1),
        (2, 2, 0, 1),
        (2, 2, 0, 1),
    )


def test_reference_reduced_shard_supports_zero_logical_numel() -> None:
    intent = make_intent(
        output=OutputSemantics.REDUCED_SHARD,
        tensor_shape=(2, 0, 3),
    )

    results = ReferenceBackend().execute_group(
        intent,
        rank_values=((), (), (), ()),
    )

    assert tuple(result.value for result in results) == ((), (), (), ())
    assert all(result.metadata.offset == 0 for result in results)
    assert all(result.metadata.stop == 0 for result in results)
    assert all(result.metadata.valid_length == 0 for result in results)
    assert all(result.metadata.padded_length == 0 for result in results)


@pytest.mark.parametrize(
    ("rank_values", "message"),
    [
        ([(1.0, 2.0)] * 4, "tuple"),
        (TupleSubclass(((1.0, 2.0),) * 4), "tuple"),
        (((1.0, 2.0),) * 3, "rank count"),
        (((1.0,),) * 4, "tensor length"),
        (((1.0, 2.0, 3.0),) * 4, "tensor length"),
        (((1.0, 2.0), [3.0, 4.0]) * 2, "rank value"),
        ((TupleSubclass((1.0, 2.0)),) * 4, "rank value"),
        (((1, 2.0),) * 4, "Python float"),
        (((True, 2.0),) * 4, "Python float"),
        (((FloatSubclass(1.0), 2.0),) * 4, "Python float"),
        (((float("nan"), 2.0),) * 4, "finite"),
        (((float("inf"), 2.0),) * 4, "finite"),
        (((float("-inf"), 2.0),) * 4, "finite"),
    ],
)
def test_reference_rejects_malformed_all_rank_inputs(
    rank_values: object,
    message: str,
) -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)

    with pytest.raises(ExecutionError, match=message):
        ReferenceBackend().execute_group(
            intent,
            cast(tuple[tuple[float, ...], ...], rank_values),
        )


def test_reference_rejects_one_wrong_length_without_zip_truncation() -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)

    with pytest.raises(ExecutionError, match="rank 2.*tensor length"):
        ReferenceBackend().execute_group(
            intent,
            rank_values=(
                (1.0, 2.0),
                (3.0, 4.0),
                (5.0,),
                (7.0, 8.0),
            ),
        )


def test_reference_rejects_nonfinite_reduction_from_finite_inputs() -> None:
    intent = make_intent(
        output=OutputSemantics.FULL_TENSOR,
        tensor_shape=(1,),
        world_size=2,
    )

    with pytest.raises(ExecutionError, match="non-finite reduction"):
        ReferenceBackend().execute_group(
            intent,
            rank_values=((1e308,), (1e308,)),
        )


def test_reference_rejects_nonexact_intent_contract() -> None:
    base = make_intent(output=OutputSemantics.FULL_TENSOR)
    intent = IntentSubclass(
        tensor=base.tensor,
        shape_family=base.shape_family,
        reduction=base.reduction,
        output=base.output,
        completion=base.completion,
        world_size=base.world_size,
        rank=base.rank,
    )

    with pytest.raises(CompileError, match="CommunicationIntent"):
        ReferenceBackend().execute_group(
            cast(CommunicationIntent, intent),
            ((1.0, 2.0),) * 4,
        )


def test_reference_oracle_cannot_register_as_a_production_backend() -> None:
    backend = ReferenceBackend()

    assert not hasattr(backend, "capabilities")
    assert not hasattr(backend, "lower")
    with pytest.raises(CompileError, match="capabilities"):
        BackendRegistry([backend])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "output",
    [OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD],
)
def test_compile_group_returns_group_only_plan_with_completed_work(
    output: OutputSemantics,
) -> None:
    backend = ReferenceBackend()
    intent = make_intent(output=output)
    strategy = native_strategy()

    plan = backend.compile_group(intent, strategy)
    work = plan.execute_group(rank_values=((1.0, 2.0),) * 4)

    assert plan.intent == intent
    assert plan.intent is not intent
    assert plan.strategy == strategy
    assert plan.strategy is not strategy
    assert not hasattr(plan, "execute")
    assert work.is_completed() is True
    results = work.result()
    if output is OutputSemantics.FULL_TENSOR:
        assert tuple(result.value for result in results) == ((4.0, 8.0),) * 4
    else:
        assert tuple(result.value for result in results) == (
            (4.0,),
            (8.0,),
            (0.0,),
            (0.0,),
        )


def test_compile_group_snapshots_complete_plan_semantics() -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    strategy = native_strategy()

    plan = ReferenceBackend().compile_group(intent, strategy)

    assert plan.intent == intent
    assert plan.intent is not intent
    assert plan.intent.tensor is not intent.tensor
    assert plan.intent.tensor.shape is not intent.tensor.shape
    assert plan.intent.shape_family is not intent.shape_family
    assert plan.strategy == strategy
    assert plan.strategy is not strategy


@pytest.mark.parametrize(
    "scenario",
    [
        "tensor-shape",
        "tensor-dtype",
        "tensor-dtype-subclass",
        "shape-family",
        "world-size",
        "rank",
        "output",
        "reduction",
    ],
)
@pytest.mark.parametrize("entrypoint", ["compile", "direct", "plan"])
def test_reference_entrypoints_revalidate_complete_intent_graphs(
    entrypoint: str,
    scenario: str,
) -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    backend = ReferenceBackend()
    if entrypoint == "plan":
        plan = backend.compile_group(intent, native_strategy())
        _forge_intent(plan.intent, scenario)
    else:
        _forge_intent(intent, scenario)

    with pytest.raises(CompileError):
        if entrypoint == "compile":
            backend.compile_group(intent, native_strategy())
        elif entrypoint == "direct":
            backend.execute_group(intent, ((1.0, 2.0),) * 4)
        else:
            plan.execute_group(((1.0, 2.0),) * 4)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("compression", object()),
        ("collective", object()),
        ("topology", object()),
        ("group_size", 0),
        ("accumulation_dtype", object()),
        ("error_feedback", 0),
        ("parameter_error_feedback", 0),
        ("overlap", 0),
        ("workspace_budget_bytes", -1),
    ],
)
@pytest.mark.parametrize("entrypoint", ["compile", "plan"])
def test_reference_compile_and_plan_revalidate_strategy_graph(
    entrypoint: str,
    field: str,
    invalid: object,
) -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    strategy = native_strategy()
    backend = ReferenceBackend()
    if entrypoint == "plan":
        plan = backend.compile_group(intent, strategy)
        object.__setattr__(plan.strategy, field, invalid)
    else:
        object.__setattr__(strategy, field, invalid)

    with pytest.raises(CompileError):
        if entrypoint == "compile":
            backend.compile_group(intent, strategy)
        else:
            plan.execute_group(((1.0, 2.0),) * 4)


def test_group_plan_validates_its_graph_before_rank_value_traversal() -> None:
    plan = ReferenceBackend().compile_group(
        make_intent(output=OutputSemantics.FULL_TENSOR),
        native_strategy(),
    )
    object.__setattr__(plan.intent.tensor, "shape", ())

    with pytest.raises(CompileError):
        plan.execute_group(object())


def test_group_plan_rejects_nonexact_plan_and_strategy_graphs() -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    strategy = native_strategy()
    subclass_strategy = StrategySubclass(
        compression=strategy.compression,
        collective=strategy.collective,
        topology=strategy.topology,
    )
    forged_strategy_plan = ReferenceBackend().compile_group(intent, strategy)
    object.__setattr__(forged_strategy_plan, "strategy", subclass_strategy)
    subclass_plan = PlanSubclass(intent=intent, strategy=strategy)

    with pytest.raises(CompileError):
        forged_strategy_plan.execute_group(((1.0, 2.0),) * 4)
    with pytest.raises(CompileError):
        subclass_plan.execute_group(((1.0, 2.0),) * 4)


def test_reference_normalizes_unexpected_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    backend = ReferenceBackend()
    plan = backend.compile_group(intent, native_strategy())

    def explode(*args: object) -> object:
        del args
        raise ValueError("unexpected oracle failure")

    monkeypatch.setattr(reference_module, "_reduce_values", explode)

    with pytest.raises(ExecutionError) as direct:
        backend.execute_group(intent, ((1.0, 2.0),) * 4)
    work = plan.execute_group(((1.0, 2.0),) * 4)

    assert type(direct.value.__cause__) is ValueError
    assert type(work) is FailedWork
    assert type(work.failure.__cause__) is ValueError
    for operation in (work.wait, work.result):
        with pytest.raises(ExecutionError) as caught:
            operation()
        assert caught.value is work.failure


def test_group_plan_propagates_execution_errors_through_failed_work() -> None:
    backend = ReferenceBackend()
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    plan = backend.compile_group(intent, native_strategy())

    work = plan.execute_group(((1.0,),) * 4)

    assert type(work) is FailedWork
    for operation in (work.wait, work.result):
        with pytest.raises(ExecutionError, match="tensor length") as caught:
            operation()
        assert caught.value is work.failure


def test_compile_group_rejects_async_intent() -> None:
    intent = make_intent(
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
    )

    with pytest.raises(CompileError, match="synchronous"):
        ReferenceBackend().compile_group(intent, native_strategy())


def test_reference_oracle_does_not_claim_declared_dtype_rounding() -> None:
    intent = make_intent(
        output=OutputSemantics.FULL_TENSOR,
        dtype="float32",
    )

    plan = ReferenceBackend().compile_group(intent, native_strategy())
    results = plan.execute_group(((0.1, 0.2),) * 4).result()

    assert tuple(result.value for result in results) == ((0.4, 0.8),) * 4


@pytest.mark.parametrize("dimension", ["compression", "collective"])
def test_compile_group_rejects_non_native_collective_strategy(
    dimension: str,
) -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    strategy = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
    )

    with pytest.raises(CompileError, match=dimension):
        ReferenceBackend().compile_group(intent, strategy)


@pytest.mark.parametrize("topology", [TopologyKind.RING, TopologyKind.TREE])
def test_compile_group_rejects_explicit_topology(
    topology: TopologyKind,
) -> None:
    strategy = StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=topology,
    )

    with pytest.raises(CompileError, match="topology"):
        ReferenceBackend().compile_group(
            make_intent(output=OutputSemantics.FULL_TENSOR),
            strategy,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accumulation_dtype", AccumulationDType.FP16, "accumulation"),
        ("error_feedback", True, "error feedback"),
        (
            "parameter_error_feedback",
            True,
            "parameter error feedback",
        ),
        ("overlap", True, "overlap"),
        ("workspace_budget_bytes", 1, "workspace"),
    ],
)
def test_compile_group_rejects_each_unimplemented_strategy_field(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "compression": CompressionKind.NONE,
        "collective": CollectiveKind.NATIVE,
        "topology": TopologyKind.BACKEND_DEFAULT,
        "group_size": None,
        "accumulation_dtype": AccumulationDType.FP32,
        "error_feedback": False,
        "parameter_error_feedback": False,
        "overlap": False,
        "workspace_budget_bytes": None,
    }
    values[field] = value
    strategy = StrategySpec(**values)  # type: ignore[arg-type]

    with pytest.raises(CompileError, match=message):
        ReferenceBackend().compile_group(
            make_intent(output=OutputSemantics.FULL_TENSOR),
            strategy,
        )


def test_none_group_size_is_rejected_before_reference_compilation() -> None:
    with pytest.raises(CompileError, match="group size"):
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=TopologyKind.BACKEND_DEFAULT,
            group_size=128,
        )


@pytest.mark.parametrize("argument", [object(), IntentSubclass])
def test_compile_group_rejects_nonexact_compile_inputs(
    argument: object,
) -> None:
    backend = ReferenceBackend()
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)

    if argument is IntentSubclass:
        invalid_intent: object = IntentSubclass(
            tensor=intent.tensor,
            shape_family=intent.shape_family,
            reduction=intent.reduction,
            output=intent.output,
            completion=intent.completion,
            world_size=intent.world_size,
            rank=intent.rank,
        )
        invalid_strategy: object = native_strategy()
    else:
        invalid_intent = intent
        invalid_strategy = argument

    with pytest.raises(CompileError):
        backend.compile_group(
            cast(CommunicationIntent, invalid_intent),
            cast(StrategySpec, invalid_strategy),
        )
