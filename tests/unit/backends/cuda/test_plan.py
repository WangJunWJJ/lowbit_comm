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
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.api.result import ReducedShardMetadata, ReducedShardResult
from lowbit_comm.backends.cuda.layout import (
    build_fulltensor_layout,
    build_reduced_shard_layout,
)
from lowbit_comm.backends.cuda.plan import (
    CudaBackendPlan,
    CudaReducedShardPlan,
)
from lowbit_comm.core.errors import CompileError, ExecutionError


def _intent() -> CommunicationIntent:
    return CommunicationIntent(
        tensor=TensorSpec(dtype="fp16", shape=(10,)),
        shape_family=ShapeFamily(max_numel=64, alignment=1),
        reduction=ReductionOp.SUM,
        output=OutputSemantics.REDUCED_SHARD,
        completion=CompletionMode.ASYNC,
        world_size=4,
        rank=3,
    )


def _strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_REDUCE_SCATTER,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=16,
    )


def _native_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )


def _fulltensor_intent() -> CommunicationIntent:
    return CommunicationIntent(
        tensor=TensorSpec(dtype="fp16", shape=(10,)),
        shape_family=ShapeFamily(max_numel=64, alignment=1),
        reduction=ReductionOp.SUM,
        output=OutputSemantics.FULL_TENSOR,
        completion=CompletionMode.ASYNC,
        world_size=4,
        rank=3,
    )


def _metadata() -> ReducedShardMetadata:
    return ReducedShardMetadata(
        global_shape=(10,),
        offset=9,
        valid_length=1,
        padded_length=3,
        owner_rank=3,
    )


def _plan(native_plan: object) -> CudaReducedShardPlan:
    return CudaReducedShardPlan(
        _intent(),
        _strategy(),
        build_reduced_shard_layout(
            numel=10,
            dtype="fp16",
            world_size=4,
            compression=CompressionKind.INT8,
            group_size=16,
            rank=3,
        ),
        _metadata(),
        native_plan,
    )


def test_concrete_cuda_plans_reject_the_opposite_output_semantics() -> None:
    class NativePlan:
        def execute(self, value: object) -> object:
            return value

    with pytest.raises(CompileError, match="full-tensor output"):
        CudaBackendPlan(
            _intent(),
            _native_strategy(),
            build_fulltensor_layout(
                numel=10,
                dtype="fp16",
                world_size=4,
                compression=CompressionKind.NONE,
                group_size=None,
            ),
            NativePlan(),
        )
    with pytest.raises(CompileError, match="reduced-shard output"):
        CudaReducedShardPlan(
            _fulltensor_intent(),
            _native_strategy(),
            build_reduced_shard_layout(
                numel=10,
                dtype="fp16",
                world_size=4,
                compression=CompressionKind.NONE,
                group_size=None,
                rank=3,
            ),
            _metadata(),
            NativePlan(),
        )


def test_reduced_shard_plan_owns_independent_input_snapshots() -> None:
    value = object()

    class NativeWork:
        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            return value

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return NativeWork()

    requested_intent = _intent()
    requested_strategy = _strategy()
    requested_layout = build_reduced_shard_layout(
        numel=10,
        dtype="fp16",
        world_size=4,
        compression=CompressionKind.INT8,
        group_size=16,
        rank=3,
    )
    requested_metadata = _metadata()
    plan = CudaReducedShardPlan(
        requested_intent,
        requested_strategy,
        requested_layout,
        requested_metadata,
        NativePlan(),
    )

    assert plan.intent == requested_intent
    assert plan.intent is not requested_intent
    assert plan.intent.tensor is not requested_intent.tensor
    assert plan.intent.shape_family is not requested_intent.shape_family
    assert plan.strategy == requested_strategy
    assert plan.strategy is not requested_strategy
    assert plan.layout == requested_layout
    assert plan.layout is not requested_layout
    assert plan.metadata == requested_metadata
    assert plan.metadata is not requested_metadata

    object.__setattr__(requested_intent.tensor, "dtype", "bf16")
    object.__setattr__(requested_intent.shape_family, "max_numel", 0)
    object.__setattr__(requested_strategy, "group_size", 32)
    object.__setattr__(requested_layout, "offset", 0)
    object.__setattr__(requested_metadata, "offset", 0)

    assert plan.intent.tensor.dtype == "fp16"
    assert plan.intent.shape_family.max_numel == 64
    assert plan.strategy.group_size == 16
    assert plan.layout.offset == 9
    assert plan.metadata.offset == 9
    result = plan.execute("input").wait()
    assert result.value is value
    assert result.metadata is plan.metadata
    assert result.metadata.offset == 9


def test_reduced_shard_work_caches_one_exact_terminal_result() -> None:
    value = object()

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            self.wait_calls += 1
            return value

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return native_work

    native_work = NativeWork()
    work = _plan(NativePlan()).execute("input")

    assert work.is_completed() is False
    first = work.wait()
    second = work.wait()
    third = work.result()
    assert type(first) is ReducedShardResult
    assert first.value is value
    assert first.metadata == _metadata()
    assert second is first
    assert third is first
    assert native_work.wait_calls == 1


def test_reduced_shard_work_preserves_native_error_identity() -> None:
    failure = ExecutionError("native failure")

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            self.wait_calls += 1
            raise failure

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return native_work

    native_work = NativeWork()
    work = _plan(NativePlan()).execute("input")

    with pytest.raises(ExecutionError) as first:
        work.wait()
    with pytest.raises(ExecutionError) as second:
        work.result()
    assert first.value is failure
    assert second.value is failure
    assert native_work.wait_calls == 1
