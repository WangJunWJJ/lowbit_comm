import pytest

from ccdl_comm import ExecutionCounters, ExecutionInfo
from ccdl_comm.communication.cuda_completion import CudaCompletionManager, NoopCompletion
from ccdl_comm.cuda.loader import CudaExtensionStatus


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
PYTHON_FALLBACK = CudaExtensionStatus(False, None, "native work unavailable")


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

        def wait(self, stream=None):
            calls.append(("wait", stream))

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
    completion.wait_stream("target-stream")
    completion.synchronize()

    assert calls == ["record", ("wait", None), ("wait", "target-stream"), "synchronize"]


def test_manager_uses_noop_completion_for_non_cuda_tensor() -> None:
    class FakeTensor:
        is_cuda = False

    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=PYTHON_FALLBACK,
    )

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

    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=PYTHON_FALLBACK,
    )
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
    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=PYTHON_FALLBACK,
    )

    work = manager.create_work(result=3, resources=(resource,))

    assert resource in work.resources
    assert work.wait() == 3
    assert resource in work.resources


def test_result_work_caches_callback_error() -> None:
    calls = []
    error = RuntimeError("post-processing failed")
    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=PYTHON_FALLBACK,
    )

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
    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=PYTHON_FALLBACK,
    )
    work = manager.create_work(
        result=None,
        handle=type("Ready", (), {"is_completed": lambda self: True, "wait": lambda self: None})(),
        complete=lambda: callbacks.append("complete") or 5,
        execution_info=INFO,
        execution_counters=counters,
    )

    assert work.query() is False
    assert callbacks == []
    assert work.execution_info.fast_path == "python_fallback"
    assert work.execution_info.requested_strategy == INFO.requested_strategy
    assert work.wait() == 5


def test_manager_prefers_native_cuda_work_when_extension_exports_executor() -> None:
    calls = []
    native_work = object()

    class NativeExecutor:
        def run(self, result, handle, completion, resources, complete):
            calls.append((result, handle, completion, resources, complete))
            return native_work

    module = type(
        "NativeModule",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(NativeExecutor),
        },
    )()
    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=CudaExtensionStatus(True, module),
    )
    handle = object()
    completion = object()
    resource = object()

    def callback():
        return "done"

    work = manager.create_work(
        result="pending",
        handle=handle,
        completion=completion,
        resources=(resource,),
        complete=callback,
    )

    assert work is native_work
    assert calls == [("pending", handle, completion, [resource], callback)]


def test_managers_reuse_stateless_native_executor_for_same_extension() -> None:
    factory_calls = []

    class NativeExecutor:
        def run(self, result, handle, completion, resources, complete):
            return result

    def create_executor():
        factory_calls.append("create")
        return NativeExecutor()

    module = type(
        "ReusableNativeModule",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(create_executor),
        },
    )()
    status = CudaExtensionStatus(True, module)

    first = CudaCompletionManager(extension_status=status)
    second = CudaCompletionManager(extension_status=status)

    assert first.create_work(result=1, handle=object(), complete=lambda: 1) == 1
    assert second.create_work(result=2, handle=object(), complete=lambda: 2) == 2
    assert factory_calls == ["create"]


def test_declared_native_abi_does_not_silently_hide_runtime_type_errors() -> None:
    class NativeExecutor:
        def run(self, result, handle, completion, resources, complete):
            raise TypeError("native ABI contract violated")

    module = type(
        "InvalidNativeModule",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(NativeExecutor),
        },
    )()
    manager = CudaCompletionManager(
        extension_status=CudaExtensionStatus(True, module)
    )

    with pytest.raises(TypeError, match="native ABI contract violated"):
        manager.create_work(result=1, handle=object(), complete=lambda: 1)


def test_manager_keeps_immediate_generic_work_on_lower_overhead_python_path() -> None:
    class NativeExecutor:
        def run(self, result, handle, completion, resources, complete):
            raise AssertionError("native executor should not run")

    module = type(
        "NativeModule",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(NativeExecutor),
        },
    )()
    manager = CudaCompletionManager(
        extension_status=CudaExtensionStatus(True, module)
    )

    work = manager.create_work(result=1, handle=object())

    assert work.wait() == 1


def test_manager_uses_python_work_and_marks_fallback_when_native_work_is_missing() -> None:
    manager = CudaCompletionManager(
        torch_provider=lambda: None,
        extension_status=CudaExtensionStatus(True, object()),
    )

    work = manager.create_work(result=3, execution_info=INFO)

    assert work.wait() == 3
    assert work.execution_info.fast_path == "python_fallback"
