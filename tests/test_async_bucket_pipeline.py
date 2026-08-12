import pytest

from ccdl_comm.communication.async_pipeline import AsyncBucketPipeline
from ccdl_comm.communication.gather_reduce import GatheredPayloads


class FakeFuture:
    def __init__(self) -> None:
        self.result = None
        self.exception = None

    def set_result(self, result):
        self.result = result

    def set_exception(self, exception):
        self.exception = exception


class FakeInnerFuture:
    def __init__(self, calls) -> None:
        self._calls = calls

    def then(self, callback):
        self._calls.append("then")
        return callback(self)


class FakeWork:
    def __init__(self, calls) -> None:
        self._calls = calls

    def get_future(self):
        self._calls.append("get_future")
        return FakeInnerFuture(self._calls)

    def wait(self):
        self._calls.append("wait")
        return GatheredPayloads(payloads=["rank0", "rank1"], world_size=2)


class FakeCompletion:
    def __init__(self, calls) -> None:
        self._calls = calls

    def wait(self):
        self._calls.append("completion_wait")

    def wait_stream(self, stream):
        self._calls.append(("completion_wait_stream", stream))

    def synchronize(self):
        self._calls.append("completion_synchronize")


class FakeCompletionManager:
    def __init__(self, calls) -> None:
        self._calls = calls

    def record_for(self, tensor):
        self._calls.append(("record", tensor))
        return FakeCompletion(self._calls)


def test_async_pipeline_orders_gather_reduce_feedback_on_consumer_stream() -> None:
    calls = []
    outer = FakeFuture()
    work = FakeWork(calls)
    manager = FakeCompletionManager(calls)

    pipeline = AsyncBucketPipeline(
        gather_work=work,
        future=outer,
        dequantize_reduce=lambda gathered: calls.append(("reduce", gathered.payloads)) or "restored",
        update_feedback=lambda restored: calls.append(("feedback", restored)),
        advance_policy=lambda: calls.append("advance"),
        completion_manager=manager,
        consumer_stream="backward-stream",
    )

    returned = pipeline.run()

    assert returned is outer
    assert outer.result == "restored"
    assert calls == [
        "get_future",
        "then",
        "wait",
        ("reduce", ["rank0", "rank1"]),
        ("feedback", "restored"),
        "advance",
        ("record", "restored"),
        ("completion_wait_stream", "backward-stream"),
    ]


def test_async_pipeline_can_skip_cpu_completion_synchronize() -> None:
    calls = []
    outer = FakeFuture()
    work = FakeWork(calls)
    manager = FakeCompletionManager(calls)

    pipeline = AsyncBucketPipeline(
        gather_work=work,
        future=outer,
        dequantize_reduce=lambda gathered: calls.append(("reduce", gathered.payloads)) or "restored",
        update_feedback=lambda restored: calls.append(("feedback", restored)),
        advance_policy=lambda: calls.append("advance"),
        completion_manager=manager,
        synchronize_completion=False,
    )

    returned = pipeline.run()

    assert returned is outer
    assert outer.result == "restored"
    assert calls == [
        "get_future",
        "then",
        "wait",
        ("reduce", ["rank0", "rank1"]),
        ("feedback", "restored"),
        "advance",
        ("record", "restored"),
        ("completion_wait_stream", None),
    ]


def test_async_pipeline_sets_exception_on_outer_future_when_callback_fails() -> None:
    outer = FakeFuture()

    pipeline = AsyncBucketPipeline(
        gather_work=FakeWork([]),
        future=outer,
        dequantize_reduce=lambda gathered: (_ for _ in ()).throw(RuntimeError("boom")),
        update_feedback=lambda restored: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager([]),
    )

    pipeline.run()

    assert isinstance(outer.exception, RuntimeError)
    assert str(outer.exception) == "boom"


def test_async_pipeline_reraises_callback_exception_without_set_exception() -> None:
    class FutureWithoutSetException:
        def set_result(self, result):
            raise AssertionError("should not set a result")

    pipeline = AsyncBucketPipeline(
        gather_work=FakeWork([]),
        future=FutureWithoutSetException(),
        dequantize_reduce=lambda gathered: (_ for _ in ()).throw(RuntimeError("boom")),
        update_feedback=lambda restored: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager([]),
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()
