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
from lowbit_comm.backends.cuda.layout import build_reduced_shard_layout
from lowbit_comm.backends.cuda.plan import CudaReducedShardPlan
from lowbit_comm.core.errors import ExecutionError


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


def test_reduced_shard_work_reraises_the_native_execution_error_by_identity() -> None:
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
