from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Condition, Event, Lock
from time import sleep
from typing import Callable

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
from lowbit_comm.backends.cuda.backend import CudaBackend
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


def _gradient_feedback_strategy(
    output: OutputSemantics,
) -> StrategySpec:
    collective = (
        CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
        if output is OutputSemantics.FULL_TENSOR
        else CollectiveKind.COMPRESSED_REDUCE_SCATTER
    )
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=collective,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=64,
        error_feedback=True,
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


def _feedback_plan(
    output: OutputSemantics,
    native_plan: object,
) -> CudaBackendPlan | CudaReducedShardPlan:
    strategy = _gradient_feedback_strategy(output)
    if output is OutputSemantics.FULL_TENSOR:
        return CudaBackendPlan(
            _fulltensor_intent(),
            strategy,
            build_fulltensor_layout(
                numel=10,
                dtype="fp16",
                world_size=4,
                compression=CompressionKind.INT8,
                group_size=64,
            ),
            native_plan,
        )
    return CudaReducedShardPlan(
        _intent(),
        strategy,
        build_reduced_shard_layout(
            numel=10,
            dtype="fp16",
            world_size=4,
            compression=CompressionKind.INT8,
            group_size=64,
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


def test_reduced_shard_work_concurrently_publishes_one_result() -> None:
    value = object()
    start = Barrier(8)
    counter_lock = Lock()

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            with counter_lock:
                self.wait_calls += 1
            sleep(0.05)
            return value

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return native_work

    native_work = NativeWork()
    work = _plan(NativePlan()).execute("input")

    def wait_once(_: int) -> ReducedShardResult:
        start.wait()
        return work.wait()

    with ThreadPoolExecutor(max_workers=8) as threads:
        results = list(threads.map(wait_once, range(8)))

    assert all(result is results[0] for result in results)
    assert results[0].value is value
    assert native_work.wait_calls == 1
    assert work.is_completed() is True


def test_reduced_shard_work_concurrently_preserves_failure_identity() -> None:
    failure = ExecutionError("native concurrent failure")
    start = Barrier(8)
    counter_lock = Lock()

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            with counter_lock:
                self.wait_calls += 1
            sleep(0.05)
            raise failure

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return native_work

    native_work = NativeWork()
    work = _plan(NativePlan()).execute("input")

    def capture_failure(_: int) -> ExecutionError:
        start.wait()
        try:
            work.wait()
        except ExecutionError as error:
            return error
        raise AssertionError("wait unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=8) as threads:
        failures = list(threads.map(capture_failure, range(8)))

    assert all(observed is failure for observed in failures)
    assert native_work.wait_calls == 1
    assert work.is_completed() is True


def test_reduced_shard_is_completed_does_not_block_on_wait_leader() -> None:
    wait_started = Event()
    release_wait = Event()

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            self.wait_calls += 1
            wait_started.set()
            assert release_wait.wait(timeout=2.0)
            return "value"

    class NativePlan:
        def execute(self, received: object) -> NativeWork:
            assert received == "input"
            return native_work

    native_work = NativeWork()
    work = _plan(NativePlan()).execute("input")

    with ThreadPoolExecutor(max_workers=2) as threads:
        waiter = threads.submit(work.wait)
        assert wait_started.wait(timeout=2.0)
        completion = threads.submit(work.is_completed)
        assert completion.result(timeout=1.0) is False
        release_wait.set()
        assert waiter.result(timeout=2.0).value == "value"

    assert work.is_completed() is True
    assert native_work.wait_calls == 1


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_prepares_from_committed_and_commits_after_wait(
    output: OutputSemantics,
) -> None:
    first_candidate = object()
    second_candidate = object()
    launches: list[tuple[object, object | None]] = []
    candidates = iter((first_candidate, second_candidate))

    class NativeWork:
        def __init__(self, candidate: object) -> None:
            self._candidate = candidate

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            return "reduced"

        def _candidate_gradient_residual(self) -> object:
            return self._candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            launches.append((gradient, committed_residual))
            return NativeWork(next(candidates))

    plan = _feedback_plan(output, NativePlan())
    first = plan.execute("gradient-1")

    assert launches == [("gradient-1", None)]
    assert first._candidate_residual is first_candidate
    assert plan._committed_residual is None
    first.wait()
    assert plan._committed_residual is first_candidate

    second = plan.execute("gradient-2")
    assert launches[-1] == ("gradient-2", first_candidate)
    assert second._candidate_residual is second_candidate
    assert plan._committed_residual is first_candidate
    second.wait()
    assert plan._committed_residual is second_candidate


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
@pytest.mark.parametrize(
    "failure_stage",
    (
        "quant",
        "transport",
        "dequant",
        "event-record",
        "event-sync",
    ),
)
def test_gradient_feedback_failure_preserves_old_residual_identity(
    output: OutputSemantics,
    failure_stage: str,
) -> None:
    previous = object()
    rejected_candidate = object()
    failure = ExecutionError(f"{failure_stage} failure")
    fail = False

    class NativeWork:
        def __init__(self, candidate: object) -> None:
            self._candidate = candidate

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            if fail and failure_stage == "event-sync":
                raise failure
            return "reduced"

        def _candidate_gradient_residual(self) -> object:
            return self._candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient
            if fail:
                assert committed_residual is previous
                if failure_stage != "event-sync":
                    raise failure
                return NativeWork(rejected_candidate)
            assert committed_residual is None
            return NativeWork(previous)

    plan = _feedback_plan(output, NativePlan())
    plan.execute("bootstrap").wait()
    assert plan._committed_residual is previous
    fail = True

    if failure_stage == "event-sync":
        work = plan.execute("gradient")
        assert work._candidate_residual is rejected_candidate
        with pytest.raises(ExecutionError) as observed:
            work.wait()
    else:
        with pytest.raises(ExecutionError) as observed:
            plan.execute("gradient")
    assert observed.value is failure
    assert plan._committed_residual is previous


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_rejects_second_in_flight_execute(
    output: OutputSemantics,
) -> None:
    candidate = object()

    class NativeWork:
        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            return "reduced"

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return NativeWork()

    plan = _feedback_plan(output, NativePlan())
    work = plan.execute("first")

    with pytest.raises(ExecutionError, match="in-flight"):
        plan.execute("second")
    assert plan._committed_residual is None

    work.wait()
    assert plan._committed_residual is candidate
    plan.execute("third").wait()


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_concurrent_wait_commits_candidate_once(
    output: OutputSemantics,
) -> None:
    candidate = object()
    start = Barrier(8)
    counter_lock = Lock()

    class NativeWork:
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            with counter_lock:
                self.wait_calls += 1
            sleep(0.05)
            return "reduced"

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return native_work

    native_work = NativeWork()
    plan = _feedback_plan(output, NativePlan())
    work = plan.execute("gradient")

    def wait_once(_: int) -> object:
        start.wait()
        return work.wait()

    with ThreadPoolExecutor(max_workers=8) as threads:
        results = list(threads.map(wait_once, range(8)))

    assert all(result is results[0] for result in results)
    assert native_work.wait_calls == 1
    assert plan._committed_residual is candidate


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_concurrent_wait_failure_preserves_old_residual(
    output: OutputSemantics,
) -> None:
    previous = object()
    rejected_candidate = object()
    failure = ExecutionError("concurrent wait failure")
    start = Barrier(8)
    counter_lock = Lock()
    fail = False

    class NativeWork:
        wait_calls = 0

        def __init__(self, candidate: object) -> None:
            self._candidate = candidate

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            with counter_lock:
                self.wait_calls += 1
            if fail:
                sleep(0.05)
                raise failure
            return "reduced"

        def _candidate_gradient_residual(self) -> object:
            return self._candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient
            assert committed_residual is (previous if fail else None)
            return NativeWork(rejected_candidate if fail else previous)

    plan = _feedback_plan(output, NativePlan())
    plan.execute("bootstrap").wait()
    fail = True
    work = plan.execute("gradient")

    def wait_once(_: int) -> BaseException:
        start.wait()
        try:
            work.wait()
        except BaseException as observed:
            return observed
        raise AssertionError("concurrent wait unexpectedly succeeded")

    with ThreadPoolExecutor(max_workers=8) as threads:
        failures = list(threads.map(wait_once, range(8)))

    assert all(observed is failure for observed in failures)
    assert work._wait.__self__.wait_calls == 1
    assert plan._committed_residual is previous


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_pending_result_is_nonblocking_and_preserves_state(
    output: OutputSemantics,
) -> None:
    candidate = object()
    pending = ExecutionError("native result is pending")

    class NativeWork:
        result_calls = 0
        wait_calls = 0

        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            self.wait_calls += 1
            raise AssertionError("pending result must not call wait")

        def result(self) -> object:
            self.result_calls += 1
            raise pending

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return native_work

    native_work = NativeWork()
    plan = _feedback_plan(output, NativePlan())
    work = plan.execute("gradient")

    with pytest.raises(ExecutionError) as observed:
        work.result()

    assert observed.value is pending
    assert native_work.result_calls == 1
    assert native_work.wait_calls == 0
    assert plan._committed_residual is None
    assert work.is_completed() is False


def test_fulltensor_feedback_result_failure_race_finalizes_transaction() -> None:
    previous = object()
    rejected_candidate = object()
    failure = ExecutionError("native terminal result failure")
    quarantine = ExecutionError("workspace pool is quarantined")
    result_entered = Event()
    release_result = Event()
    waiter_reached = Event()
    waiter_paths: list[str] = []
    waiter_paths_lock = Lock()

    class ObservedCondition(Condition):
        def wait(self, timeout: float | None = None) -> bool:
            with waiter_paths_lock:
                waiter_paths.append("transaction-waiter")
            waiter_reached.set()
            return super().wait(timeout)

    class NativeWork:
        result_calls = 0
        wait_calls = 0

        def __init__(self, candidate: object, *, racing: bool) -> None:
            self._candidate = candidate
            self._racing = racing
            self._completed = False

        def is_completed(self) -> bool:
            return self._completed

        def wait(self) -> object:
            self.wait_calls += 1
            if self._racing:
                with waiter_paths_lock:
                    waiter_paths.append("native-wait")
                waiter_reached.set()
                raise AssertionError(
                    "the concurrent waiter must join result finalization"
                )
            self._completed = True
            return "bootstrap"

        def result(self) -> object:
            self.result_calls += 1
            assert self._racing
            result_entered.set()
            assert release_result.wait(timeout=2.0)
            self._completed = True
            native_plan.quarantined = True
            raise failure

        def _candidate_gradient_residual(self) -> object:
            return self._candidate

    class NativePlan:
        execute_calls = 0
        quarantined = False

        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient
            self.execute_calls += 1
            if self.quarantined:
                raise quarantine
            if self.execute_calls == 1:
                assert committed_residual is None
                work = NativeWork(previous, racing=False)
            else:
                assert committed_residual is previous
                work = NativeWork(rejected_candidate, racing=True)
            launched.append(work)
            return work

    def capture_failure(operation: Callable[[], object]) -> BaseException:
        try:
            operation()
        except BaseException as observed:
            return observed
        raise AssertionError("terminal operation unexpectedly succeeded")

    launched: list[NativeWork] = []
    native_plan = NativePlan()
    plan = _feedback_plan(OutputSemantics.FULL_TENSOR, native_plan)
    plan.execute("bootstrap").wait()
    work = plan.execute("gradient")
    work._condition = ObservedCondition()

    with ThreadPoolExecutor(max_workers=2) as threads:
        result_call = threads.submit(capture_failure, work.result)
        assert result_entered.wait(timeout=2.0)
        wait_call = threads.submit(capture_failure, work.wait)
        assert waiter_reached.wait(timeout=2.0)
        release_result.set()
        observed = (
            result_call.result(timeout=2.0),
            wait_call.result(timeout=2.0),
        )

    assert waiter_paths == ["transaction-waiter"]
    assert all(item is failure for item in observed)
    assert launched[-1].result_calls == 1
    assert launched[-1].wait_calls == 0
    assert work.is_completed() is True
    assert plan._committed_residual is previous
    for operation in (work.result, work.wait):
        assert capture_failure(operation) is failure

    with pytest.raises(ExecutionError) as rejected:
        plan.execute("after-failure")
    assert rejected.value is quarantine
    assert native_plan.execute_calls == 3
    assert plan._committed_residual is previous


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_successful_result_commits_without_native_wait(
    output: OutputSemantics,
) -> None:
    candidate = object()
    value = object()

    class NativeWork:
        result_calls = 0
        wait_calls = 0

        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            self.wait_calls += 1
            raise AssertionError("completed result must use native result")

        def result(self) -> object:
            self.result_calls += 1
            return value

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return native_work

    native_work = NativeWork()
    plan = _feedback_plan(output, NativePlan())
    work = plan.execute("gradient")

    first = work.result()
    second = work.result()

    if output is OutputSemantics.FULL_TENSOR:
        assert first is value
    else:
        assert first.value is value
    assert second is first
    assert native_work.result_calls == 1
    assert native_work.wait_calls == 0
    assert plan._committed_residual is candidate


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_failed_result_repeats_identity_and_preserves_old(
    output: OutputSemantics,
) -> None:
    previous = object()
    rejected_candidate = object()
    failure = ExecutionError("native result failure")
    fail = False

    class NativeWork:
        result_calls = 0
        wait_calls = 0

        def __init__(self, candidate: object) -> None:
            self._candidate = candidate

        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            self.wait_calls += 1
            if fail:
                raise AssertionError("failed result must use native result")
            return "bootstrap"

        def result(self) -> object:
            self.result_calls += 1
            if fail:
                raise failure
            return "bootstrap"

        def _candidate_gradient_residual(self) -> object:
            return self._candidate

    launched: list[NativeWork] = []

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient
            assert committed_residual is (previous if fail else None)
            work = NativeWork(rejected_candidate if fail else previous)
            launched.append(work)
            return work

    plan = _feedback_plan(output, NativePlan())
    plan.execute("bootstrap").wait()
    fail = True
    work = plan.execute("gradient")

    for operation in (work.result, work.result, work.wait):
        with pytest.raises(ExecutionError) as observed:
            operation()
        assert observed.value is failure

    assert launched[-1].result_calls == 1
    assert launched[-1].wait_calls == 0
    assert plan._committed_residual is previous


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_concurrent_result_publishes_once(
    output: OutputSemantics,
) -> None:
    candidate = object()
    value = object()
    start = Barrier(8)
    counter_lock = Lock()

    class NativeWork:
        result_calls = 0

        def is_completed(self) -> bool:
            return True

        def wait(self) -> object:
            raise AssertionError("concurrent result must not call wait")

        def result(self) -> object:
            with counter_lock:
                self.result_calls += 1
            sleep(0.05)
            return value

        def _candidate_gradient_residual(self) -> object:
            return candidate

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return native_work

    native_work = NativeWork()
    plan = _feedback_plan(output, NativePlan())
    work = plan.execute("gradient")

    def result_once(_: int) -> object:
        start.wait()
        return work.result()

    with ThreadPoolExecutor(max_workers=8) as threads:
        results = list(threads.map(result_once, range(8)))

    assert all(result is results[0] for result in results)
    assert native_work.result_calls == 1
    assert plan._committed_residual is candidate


@pytest.mark.parametrize(
    "output",
    (OutputSemantics.FULL_TENSOR, OutputSemantics.REDUCED_SHARD),
)
def test_gradient_feedback_work_forwards_token_and_native_diagnostics(
    output: OutputSemantics,
) -> None:
    candidate = object()
    token = object()
    latch_state = object()
    calls: list[object] = []

    class NativeWork:
        def is_completed(self) -> bool:
            return False

        def wait(self) -> object:
            return "result"

        def result(self) -> object:
            raise ExecutionError("pending")

        def launch_token(self) -> object:
            return token

        def _candidate_gradient_residual(self) -> object:
            return candidate

        def _synchronize_count_for_test(self) -> int:
            return 7

        def _enable_wait_latch_for_test(self, expected: int) -> object:
            calls.append(("enable", expected))
            return None

        def _wait_latch_state_for_test(self) -> object:
            return latch_state

        def _allow_completion_for_test(self) -> object:
            calls.append("allow")
            return None

        def _release_losers_for_test(self) -> object:
            calls.append("release")
            return None

    class NativePlan:
        def execute(
            self,
            gradient: object,
            committed_residual: object | None,
        ) -> NativeWork:
            del gradient, committed_residual
            return NativeWork()

    work = _feedback_plan(output, NativePlan()).execute("gradient")

    assert work.launch_token() is token
    assert work._synchronize_count_for_test() == 7
    assert work._enable_wait_latch_for_test(3) is None
    assert work._wait_latch_state_for_test() is latch_state
    assert work._allow_completion_for_test() is None
    assert work._release_losers_for_test() is None
    assert calls == [("enable", 3), "allow", "release"]


@pytest.mark.parametrize(
    ("output", "strategy"),
    [
        (OutputSemantics.FULL_TENSOR, _native_strategy()),
        (OutputSemantics.FULL_TENSOR, _strategy()),
        (
            OutputSemantics.REDUCED_SHARD,
            StrategySpec(
                compression=CompressionKind.INT8,
                collective=CollectiveKind.COMPRESSED_REDUCE_SCATTER,
                topology=TopologyKind.BACKEND_DEFAULT,
                group_size=32,
                error_feedback=True,
            ),
        ),
    ],
)
def test_gradient_feedback_stays_inside_int8_group64_private_boundary(
    output: OutputSemantics,
    strategy: StrategySpec,
) -> None:
    if not strategy.error_feedback:
        strategy = StrategySpec(
            compression=strategy.compression,
            collective=strategy.collective,
            topology=strategy.topology,
            group_size=strategy.group_size,
            error_feedback=True,
        )

    with pytest.raises(CompileError, match="error feedback"):
        if output is OutputSemantics.FULL_TENSOR:
            CudaBackendPlan(
                _fulltensor_intent(),
                strategy,
                build_fulltensor_layout(
                    numel=10,
                    dtype="fp16",
                    world_size=4,
                    compression=strategy.compression,
                    group_size=strategy.group_size,
                ),
                object(),
            )
        else:
            CudaReducedShardPlan(
                _intent(),
                strategy,
                build_reduced_shard_layout(
                    numel=10,
                    dtype="fp16",
                    world_size=4,
                    compression=strategy.compression,
                    group_size=strategy.group_size,
                    rank=3,
                ),
                _metadata(),
                object(),
            )


def test_gradient_feedback_is_not_advertised_as_a_cuda_capability() -> None:
    assert all(
        not capability.strategy.error_feedback
        for capability in CudaBackend().capabilities()
    )
