"""Contract tests for deterministic all-rank reference execution."""

from dataclasses import FrozenInstanceError
from typing import cast

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
    CollectiveKind,
    CompressionKind,
    NativePolicy,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.api.result import (
    FullTensorResult,
    ReducedShardResult,
)
from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler.compiler import Compiler
from lowbit_comm.compiler.evidence import (
    EnvironmentFingerprint,
    EvidenceStore,
)
from lowbit_comm.compiler.registry import BackendRegistry
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
    ExecutionError,
)
from lowbit_comm.core.plan import CompilationContext
from lowbit_comm.runtime.work import FailedWork


ReferenceResult = (
    FullTensorResult[tuple[float, ...]]
    | ReducedShardResult[tuple[float, ...]]
)


class FloatSubclass(float):
    """A float subclass that must not cross the exact scalar boundary."""


class TupleSubclass(tuple[object, ...]):
    """A tuple subclass that must not cross a container boundary."""


class IntentSubclass(CommunicationIntent):
    """An intent subclass that must not cross the exact contract boundary."""


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
    assert [result.value for result in full_results] == [
        (4.0, 5.0)
    ] * 4
    assert len({id(result) for result in full_results}) == 4
    with pytest.raises(FrozenInstanceError):
        full_results[0].value = ()


def test_reference_fulltensor_sum_uses_logical_multidimensional_numel(
) -> None:
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
        tuple(float(rank + index) for index in range(10))
        for rank in range(4)
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
        len(result.value) == result.metadata.padded_length
        for result in shard_results
    )
    assert all(
        result.metadata.global_shape == (10,)
        and result.metadata.owner_rank == rank
        for rank, result in enumerate(shard_results)
    )


def test_reference_reduced_shard_mean_is_applied_before_partitioning(
) -> None:
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


def test_reference_reduced_shard_supports_more_ranks_than_elements(
) -> None:
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


def test_reference_advertises_only_exact_synchronous_native_contracts(
) -> None:
    backend = ReferenceBackend()

    capabilities = backend.capabilities()

    assert backend.backend_id == "reference"
    assert type(capabilities) is tuple
    assert len(capabilities) == 2
    assert {capability.output for capability in capabilities} == {
        OutputSemantics.FULL_TENSOR,
        OutputSemantics.REDUCED_SHARD,
    }
    assert all(
        capability.backend_id == "reference"
        and capability.compression is CompressionKind.NONE
        and capability.collective is CollectiveKind.NATIVE
        and capability.topology is TopologyKind.BACKEND_DEFAULT
        and capability.min_world_size == 1
        and capability.max_world_size is None
        and capability.supported_dtypes == frozenset({"float16"})
        and capability.supports_async is False
        for capability in capabilities
    )
    assert backend.capabilities() is capabilities


def test_compile_group_returns_protocol_plan_with_completed_work() -> None:
    backend = ReferenceBackend()
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)

    plan = backend.compile_group(intent, native_strategy())
    work = plan.execute(((1.0, 2.0),) * 4)

    assert work.is_completed() is True
    assert tuple(result.value for result in work.result()) == (
        (4.0, 8.0),
    ) * 4


def test_protocol_plan_propagates_execution_errors_through_failed_work(
) -> None:
    backend = ReferenceBackend()
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    plan = backend.lower(intent, native_strategy())

    work = plan.execute(((1.0,),) * 4)

    assert type(work) is FailedWork
    for operation in (work.wait, work.result):
        with pytest.raises(ExecutionError, match="tensor length"):
            operation()


def test_reference_integrates_with_registry_and_compiler() -> None:
    intent = make_intent(output=OutputSemantics.REDUCED_SHARD)
    context = CompilationContext(
        environment=EnvironmentFingerprint.from_mapping(
            {
                "accelerator": "cpu",
                "interconnect": "none",
                "software": "python",
            }
        ),
        workspace_budget_bytes=0,
        node_count=1,
        workload_class="contract",
        bucket_min_bytes=4,
        bucket_max_bytes=4,
    )

    plan = Compiler(
        BackendRegistry([ReferenceBackend()]),
        EvidenceStore(),
    ).compile(intent, NativePolicy(), context)
    work = plan.backend_plan.execute(((1.0, 2.0),) * 4)

    assert plan.backend_id == "reference"
    assert tuple(result.value for result in work.result()) == (
        (4.0,),
        (8.0,),
        (0.0,),
        (0.0,),
    )


@pytest.mark.parametrize(
    "intent",
    [
        make_intent(
            output=OutputSemantics.FULL_TENSOR,
            completion=CompletionMode.ASYNC,
        ),
        make_intent(
            output=OutputSemantics.FULL_TENSOR,
            dtype="float32",
        ),
    ],
)
def test_compile_group_rejects_unadvertised_intent(
    intent: CommunicationIntent,
) -> None:
    with pytest.raises(CapabilityError):
        ReferenceBackend().compile_group(intent, native_strategy())


def test_compile_group_rejects_quantization_and_topology_claims() -> None:
    intent = make_intent(output=OutputSemantics.FULL_TENSOR)
    strategy = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.RING,
        group_size=128,
    )

    with pytest.raises(CapabilityError):
        ReferenceBackend().compile_group(intent, strategy)


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
