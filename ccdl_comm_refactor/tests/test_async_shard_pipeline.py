from ccdl_comm.collectives.reduce_scatter import ReducedShard
from ccdl_comm.communication.async_shard_pipeline import AsyncShardPipeline


def test_async_shard_pipeline_is_exported_from_communication_package() -> None:
    from ccdl_comm.communication import AsyncShardPipeline as ExportedPipeline

    assert ExportedPipeline is AsyncShardPipeline


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
        return ["payload-rank0", "payload-rank1"]


class FakeCompletion:
    def __init__(self, calls) -> None:
        self._calls = calls

    def wait(self):
        self._calls.append("completion_wait")

    def synchronize(self):
        self._calls.append("completion_synchronize")


class FakeCompletionManager:
    def __init__(self, calls) -> None:
        self._calls = calls

    def record_for(self, tensor):
        self._calls.append(("record", tensor))
        return FakeCompletion(self._calls)


class FakeLease:
    def __init__(self, calls) -> None:
        self._calls = calls
        self.buffer = "leased-buffer"

    def release(self, *, completion) -> None:
        self._calls.append(("lease_release", completion))


def test_async_shard_pipeline_orders_work_reduce_feedback_completion_and_future() -> None:
    calls = []
    outer = FakeFuture()
    work = FakeWork(calls)
    manager = FakeCompletionManager(calls)

    pipeline = AsyncShardPipeline(
        communication_work=work,
        future=outer,
        reduce_shard=lambda payloads: calls.append(("reduce", payloads)) or _shard("reduced"),
        update_feedback=lambda shard: calls.append(("feedback", shard.shard)),
        advance_policy=lambda: calls.append("advance"),
        completion_manager=manager,
    )

    returned = pipeline.run()

    assert returned is pipeline
    assert returned.query() is True
    assert returned.wait() is outer.result
    assert returned.get_future() is outer
    assert outer.result.shard == "reduced"
    assert outer.result.metadata["async_completion"] is True
    assert calls == [
        "get_future",
        "then",
        "wait",
        ("reduce", ["payload-rank0", "payload-rank1"]),
        ("feedback", "reduced"),
        "advance",
        ("record", "reduced"),
        "completion_wait",
    ]


def test_async_shard_pipeline_can_request_cpu_completion_synchronize() -> None:
    calls = []
    outer = FakeFuture()

    pipeline = AsyncShardPipeline(
        communication_work=FakeWork(calls),
        future=outer,
        reduce_shard=lambda payloads: calls.append(("reduce", payloads)) or _shard("reduced"),
        update_feedback=lambda shard: calls.append(("feedback", shard.shard)),
        advance_policy=lambda: calls.append("advance"),
        completion_manager=FakeCompletionManager(calls),
        synchronize_completion=True,
    )

    pipeline.run()

    assert calls[-3:] == [("record", "reduced"), "completion_wait", "completion_synchronize"]


def test_async_shard_pipeline_releases_workspace_after_completion_is_ordered() -> None:
    calls = []
    lease = FakeLease(calls)
    pipeline = AsyncShardPipeline(
        communication_work=FakeWork(calls),
        future=FakeFuture(),
        reduce_shard=lambda payloads: calls.append(("reduce", payloads)) or _shard("reduced"),
        update_feedback=lambda shard: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager(calls),
        workspace_leases=(lease,),
    )

    pipeline.run()

    completion = calls[-1][1]
    assert calls[-3:] == [
        ("record", "reduced"),
        "completion_wait",
        ("lease_release", completion),
    ]


def test_async_shard_pipeline_retains_inflight_resources() -> None:
    resource = object()
    pipeline = AsyncShardPipeline(
        communication_work=FakeWork([]),
        future=FakeFuture(),
        reduce_shard=lambda payloads: _shard("reduced"),
        update_feedback=lambda shard: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager([]),
        resources=(resource,),
    )

    work = pipeline.run()

    assert work.resources == (resource,)
    assert work.wait().shard == "reduced"


def test_async_shard_pipeline_sets_exception_on_outer_future_when_callback_fails() -> None:
    outer = FakeFuture()

    pipeline = AsyncShardPipeline(
        communication_work=FakeWork([]),
        future=outer,
        reduce_shard=lambda payloads: (_ for _ in ()).throw(RuntimeError("boom")),
        update_feedback=lambda shard: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager([]),
    )

    pipeline.run()

    assert isinstance(outer.exception, RuntimeError)
    assert str(outer.exception) == "boom"


def test_async_shard_pipeline_releases_workspace_when_callback_fails() -> None:
    calls = []
    outer = FakeFuture()
    lease = FakeLease(calls)
    pipeline = AsyncShardPipeline(
        communication_work=FakeWork(calls),
        future=outer,
        reduce_shard=lambda payloads: (_ for _ in ()).throw(RuntimeError("boom")),
        update_feedback=lambda shard: None,
        advance_policy=lambda: None,
        completion_manager=FakeCompletionManager(calls),
        workspace_leases=(lease,),
    )

    pipeline.run()

    assert isinstance(outer.exception, RuntimeError)
    assert calls[-3][0] == "record"
    assert calls[-3][1] == lease.buffer
    assert calls[-2] == "completion_wait"
    assert calls[-1][0] == "lease_release"


def _shard(tensor: str) -> ReducedShard:
    return ReducedShard(
        shard=tensor,
        shard_index=0,
        shard_numel=1,
        original_shape=(2,),
        original_numel=2,
        world_size=2,
        reduce="mean",
        dtype="float16",
        metadata={},
    )
