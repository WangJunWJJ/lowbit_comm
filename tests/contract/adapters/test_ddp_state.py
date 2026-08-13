from __future__ import annotations

from concurrent.futures import Future

import pytest

from lowbit_comm.adapters.ddp import (
    CompressionSchema,
    GradientFeedbackState,
    create_ddp_hook,
)


SCHEMA = CompressionSchema(
    bit=8,
    group_size=64,
    quant_type="linear",
    compact=False,
    algorithm="compressed_rs_ag",
)


class Value:
    def __init__(self, values: tuple[float, ...], *, finite: bool = True) -> None:
        self.values = values
        self.shape = (len(values),)
        self.dtype = "fp16"
        self.finite = finite

    def __add__(self, other: Value) -> Value:
        return Value(tuple(a + b for a, b in zip(self.values, other.values)))

    def __sub__(self, other: Value) -> Value:
        return Value(tuple(a - b for a, b in zip(self.values, other.values)))

    def detach(self) -> Value:
        return self

    def clone(self) -> Value:
        return Value(self.values, finite=self.finite)


class CudaValue:
    is_cuda = True

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.shape = (len(values),)
        self.dtype = "fp16"

    @property
    def finite(self) -> bool:
        raise AssertionError("CUDA feedback validation must not inspect host state")

    def isfinite(self) -> object:
        raise AssertionError("CUDA feedback validation must not synchronize the host")

    def __sub__(self, other: CudaValue) -> Value:
        return Value(tuple(a - b for a, b in zip(self.values, other.values)))


class Bucket:
    def __init__(self, value: Value, index: int = 0) -> None:
        self._value = value
        self._index = index

    def buffer(self) -> Value:
        return self._value

    def index(self) -> int:
        return self._index


class ImmediateWork:
    def __init__(self, result: Value, calls: list[str]) -> None:
        self._result = result
        self._calls = calls

    def wait(self) -> Value:
        self._calls.append("work.wait")
        return self._result


class Executable:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def reconstruct_local(self, prepared: Value) -> Value:
        self.calls.append("reconstruct")
        return Value(tuple(value - 0.25 for value in prepared.values))

    def run(self, prepared: Value) -> ImmediateWork:
        self.calls.append("run")
        return ImmediateWork(Value((9.0, 9.0)), self.calls)


class FusedExecutable(Executable):
    def run_with_local_reconstruction(
        self, prepared: Value
    ) -> tuple[ImmediateWork, Value]:
        self.calls.append("run_fused")
        return (
            ImmediateWork(Value((9.0, 9.0)), self.calls),
            Value(tuple(value - 0.25 for value in prepared.values)),
        )


def test_feedback_commits_local_reconstruction_error_transactionally() -> None:
    state = GradientFeedbackState()
    transaction = state.prepare(
        0,
        Value((1.0, 2.0)),
        world_size=2,
        compression_schema=SCHEMA,
    )

    assert transaction.prepared.values == (1.0, 2.0)
    transaction.commit(Value((0.75, 1.75)))
    assert state.residual(0).values == (0.25, 0.25)
    assert state.prepare(
        0,
        Value((3.0, 4.0)),
        world_size=2,
        compression_schema=SCHEMA,
    ).prepared.values == (3.25, 4.25)


def test_feedback_identity_includes_world_size_and_compression_schema() -> None:
    state = GradientFeedbackState()
    state.prepare(
        0,
        Value((1.0,)),
        world_size=2,
        compression_schema=SCHEMA,
    ).commit(Value((0.5,)))
    changed_schema = CompressionSchema(
        bit=8,
        group_size=64,
        quant_type="linear",
        compact=True,
        algorithm="compressed_all_gather",
    )

    assert state.prepare(
        0,
        Value((3.0,)),
        world_size=4,
        compression_schema=SCHEMA,
    ).prepared.values == (3.0,)
    assert state.prepare(
        0,
        Value((4.0,)),
        world_size=2,
        compression_schema=changed_schema,
    ).prepared.values == (4.0,)


def test_failed_or_overflowed_transaction_never_mutates_residual() -> None:
    state = GradientFeedbackState()
    state.prepare(0, Value((1.0,)), world_size=2, compression_schema=SCHEMA).commit(
        Value((0.5,))
    )
    before = state.residual(0).values

    state.prepare(0, Value((3.0,)), world_size=2, compression_schema=SCHEMA).abort()
    overflow = state.prepare(
        0,
        Value((4.0,), finite=False),
        world_size=2,
        compression_schema=SCHEMA,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        overflow.commit(Value((4.0,)))
    assert state.residual(0).values == before


def test_cuda_feedback_does_not_run_host_synchronizing_finite_check() -> None:
    state = GradientFeedbackState()
    transaction = state.prepare(
        0,
        CudaValue((1.0, 2.0)),
        world_size=2,
        compression_schema=SCHEMA,
    )

    transaction.commit(CudaValue((0.75, 1.75)))

    assert state.residual(0).values == (0.25, 0.25)


def test_bucket_rebuild_invalidates_old_layout_generation() -> None:
    state = GradientFeedbackState(layout_generation=1)
    state.prepare(
        0,
        Value((1.0, 2.0)),
        world_size=2,
        compression_schema=SCHEMA,
    ).commit(Value((0.5, 1.5)))
    state.rebuild(layout_generation=2)

    assert state.residual(0) is None
    assert state.prepare(
        0,
        Value((3.0, 4.0)),
        world_size=2,
        compression_schema=SCHEMA,
    ).key.layout_generation == 2


def test_checkpoint_invalidation_rejects_inflight_feedback_commit() -> None:
    state = GradientFeedbackState(layout_generation=2)
    transaction = state.prepare(
        0,
        Value((1.0,)),
        world_size=2,
        compression_schema=SCHEMA,
    )

    state.invalidate("checkpoint_restore")

    assert state.last_invalidation_reason == "checkpoint_restore"
    assert state.residual(0) is None
    with pytest.raises(RuntimeError, match="obsolete feedback state"):
        transaction.commit(Value((0.5,)))


def test_feedback_invalidation_reason_must_be_non_empty() -> None:
    state = GradientFeedbackState()

    with pytest.raises(ValueError, match="reason"):
        state.invalidate("")


def test_hook_future_completes_after_work_and_feedback_commit() -> None:
    calls: list[str] = []
    state = GradientFeedbackState(on_commit=lambda: calls.append("commit"))
    hook = create_ddp_hook(
        Executable(calls),
        state=state,
        world_size=2,
        compression_schema=SCHEMA,
        future_factory=Future,
    )

    future = hook(None, Bucket(Value((1.0, 2.0))))
    result = future.result(timeout=2.0)

    assert result.values == (9.0, 9.0)
    assert calls == ["reconstruct", "run", "work.wait", "commit"]
    assert future.done()


def test_hook_reuses_communication_quantization_for_feedback() -> None:
    calls: list[str] = []
    state = GradientFeedbackState(on_commit=lambda: calls.append("commit"))
    hook = create_ddp_hook(
        FusedExecutable(calls),
        state=state,
        world_size=2,
        compression_schema=SCHEMA,
        future_factory=Future,
    )

    result = hook(None, Bucket(Value((1.0, 2.0)))).result(timeout=2.0)

    assert result.values == (9.0, 9.0)
    assert calls == ["run_fused", "work.wait", "commit"]


def test_hook_accepts_framework_runtime_annotations() -> None:
    hook = create_ddp_hook(
        Executable([]),
        state=GradientFeedbackState(),
        world_size=2,
        compression_schema=SCHEMA,
        future_factory=Future,
        bucket_type=Bucket,
        return_type=Future,
    )

    assert hook.__annotations__["bucket"] is Bucket
    assert hook.__annotations__["return"] is Future
