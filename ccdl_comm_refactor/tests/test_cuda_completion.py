from ccdl_comm import ExecutionCounters, ExecutionInfo
from ccdl_comm.communication.cuda_completion import CudaCompletionManager, NoopCompletion


INFO = ExecutionInfo(
    requested_strategy="all_gather",
    executed_strategy="all_gather",
    backend="cuda",
    fallback_used=False,
    fallback_reason=None,
    stage_names=(),
    original_bytes=2,
    compressed_bytes=1,
    compression_ratio=2.0,
    workspace_cache_hit=False,
    async_capable=True,
    fast_path="cuda_all_gather",
)


def test_completion_work_is_exported_from_public_packages() -> None:
    from ccdl_comm import CompletionWork as TopLevelCompletionWork
    from ccdl_comm.collectives import CompletionWork

    assert TopLevelCompletionWork is CompletionWork


def test_noop_completion_is_safe_without_torch() -> None:
    completion = NoopCompletion()

    completion.wait()
    completion.synchronize()


def test_manager_records_event_for_cuda_tensor_with_injected_torch() -> None:
    calls = []

    class FakeEvent:
        def record(self):
            calls.append("record")

        def wait(self):
            calls.append("wait")

        def synchronize(self):
            calls.append("synchronize")

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda

    class FakeTensor:
        is_cuda = True

    manager = CudaCompletionManager(torch_provider=lambda: FakeTorch)
    completion = manager.record_for(FakeTensor())
    completion.wait()
    completion.synchronize()

    assert calls == ["record", "wait", "synchronize"]


def test_manager_uses_noop_completion_for_non_cuda_tensor() -> None:
    class FakeTensor:
        is_cuda = False

    manager = CudaCompletionManager(torch_provider=lambda: None)

    assert isinstance(manager.record_for(FakeTensor()), NoopCompletion)


def test_manager_creates_cuda_stream_work_with_event_ordering() -> None:
    calls = []

    class FakeStream:
        def __init__(self, index):
            self.device = type("Device", (), {"index": index})()

        def wait_stream(self, stream):
            calls.append(("wait_stream", self.device.index, stream.device.index))

    class FakeStreamGuard:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            calls.append(("enter", self.stream.device.index))

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("exit", self.stream.device.index))

    class FakeEvent:
        def record(self, stream=None):
            calls.append(("record", stream.device.index if stream is not None else None))

        def wait(self):
            calls.append("event_wait")

        def query(self):
            calls.append("query")
            return True

    class FakeHandle:
        def wait(self):
            calls.append("handle_wait")

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def current_stream():
            return FakeStream(99)

        @staticmethod
        def Stream(device):
            calls.append(("new_stream", device))
            return FakeStream(device)

        @staticmethod
        def stream(stream):
            return FakeStreamGuard(stream)

    class FakeTorch:
        cuda = FakeCuda

    manager = CudaCompletionManager(torch_provider=lambda: FakeTorch)
    work = manager.create_stream_work(async_op=True, handle=FakeHandle())
    with work:
        calls.append("body")

    assert work.query() is True
    work.wait()

    assert calls == [
        ("new_stream", 0),
        ("wait_stream", 0, 99),
        ("enter", 0),
        "body",
        ("exit", 0),
        ("record", 0),
        "query",
        "handle_wait",
        "event_wait",
    ]


def test_result_work_waits_for_handle_before_completing_once() -> None:
    calls = []

    class FakeHandle:
        def wait(self):
            calls.append("handle_wait")

        def is_completed(self):
            calls.append("handle_query")
            return False

    manager = CudaCompletionManager(torch_provider=lambda: None)
    work = manager.create_work(
        result="pending",
        handle=FakeHandle(),
        complete=lambda: calls.append("complete") or "finished",
    )

    assert work.query() is False
    assert calls == ["handle_query"]
    assert work.wait() == "finished"
    assert work.wait() == "finished"
    assert calls == ["handle_query", "handle_wait", "complete"]


def test_result_work_retains_resources_until_completion() -> None:
    resource = object()
    manager = CudaCompletionManager(torch_provider=lambda: None)

    work = manager.create_work(result=3, resources=(resource,))

    assert resource in work.resources
    assert work.wait() == 3
    assert resource in work.resources


def test_result_work_caches_callback_error() -> None:
    calls = []
    error = RuntimeError("post-processing failed")
    manager = CudaCompletionManager(torch_provider=lambda: None)

    def fail():
        calls.append("complete")
        raise error

    work = manager.create_work(result=None, complete=fail)

    for _ in range(2):
        try:
            work.wait()
        except RuntimeError as exc:
            assert exc is error
        else:
            raise AssertionError("wait() did not re-raise the callback error")

    assert calls == ["complete"]


def test_manager_forwards_execution_metadata_without_query_side_effects() -> None:
    callbacks = []
    counters = ExecutionCounters()
    manager = CudaCompletionManager(torch_provider=lambda: None)
    work = manager.create_work(
        result=None,
        handle=type("Ready", (), {"is_completed": lambda self: True, "wait": lambda self: None})(),
        complete=lambda: callbacks.append("complete") or 5,
        execution_info=INFO,
        execution_counters=counters,
    )

    assert work.query() is False
    assert callbacks == []
    assert work.execution_info is INFO
    assert work.wait() == 5
