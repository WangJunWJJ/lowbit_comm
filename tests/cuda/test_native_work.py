from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

from ccdl_comm.cuda.loader import load_cuda_extension


@pytest.fixture(scope="module")
def extension():
    status = load_cuda_extension()
    if not status.available:
        pytest.skip(status.reason or "CCDL CUDA extension is unavailable")
    return status.module


@pytest.fixture(scope="module")
def cuda_extension(extension):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return extension


class _Handle:
    def __init__(self, *, completed: bool = False, error: BaseException | None = None):
        self.completed = completed
        self.error = error
        self.query_calls = 0
        self.wait_calls = 0
        self.future = object()

    def is_completed(self) -> bool:
        self.query_calls += 1
        return self.completed

    def wait(self) -> None:
        self.wait_calls += 1
        if self.error is not None:
            raise self.error
        self.completed = True

    def get_future(self):
        return self.future


class _Completion:
    def __init__(self, *, completed: bool = False):
        self.completed = completed
        self.query_calls = 0
        self.wait_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        return self.completed

    def wait(self) -> None:
        self.wait_calls += 1
        self.completed = True


def test_cuda_extension_exports_native_work(extension) -> None:
    assert hasattr(extension, "CompressedWork")
    assert hasattr(extension, "create_cuda_executor")
    assert extension.NATIVE_WORK_ABI_VERSION == 1


def test_native_work_supports_deferred_generic_result(extension) -> None:
    handle = _Handle()
    work = extension.CompressedWork(
        None,
        handle,
        None,
        [object()],
        lambda: ["rank-0", "rank-1"],
    )

    assert work.wait() == ["rank-0", "rank-1"]
    assert work.result() == ["rank-0", "rank-1"]


def test_query_is_non_blocking_and_checks_transport_before_completion(cuda_extension) -> None:
    import torch

    handle = _Handle(completed=False)
    completion = _Completion(completed=False)
    work = cuda_extension.create_cuda_executor().run(
        torch.ones(1, device="cuda"), handle, completion
    )

    assert work.query() is False
    assert handle.query_calls == 1
    assert handle.wait_calls == 0
    assert completion.query_calls == 0
    assert completion.wait_calls == 0

    handle.completed = True
    assert work.query() is False
    assert completion.query_calls == 1
    assert completion.wait_calls == 0


def test_wait_is_repeatable_and_completion_runs_once(cuda_extension) -> None:
    import torch

    result = torch.ones(4, device="cuda")
    handle = _Handle()
    completion = _Completion()
    callback_calls = 0

    def complete():
        nonlocal callback_calls
        callback_calls += 1
        return result.mul_(2)

    work = cuda_extension.CompressedWork(
        result, handle, completion, [result], complete
    )

    first = work.wait()
    second = work.wait()

    assert first.data_ptr() == result.data_ptr()
    assert second.data_ptr() == result.data_ptr()
    assert first.tolist() == [2.0] * 4
    assert handle.wait_calls == 1
    assert completion.wait_calls == 1
    assert callback_calls == 1
    assert work.query() is True
    assert work.result().data_ptr() == result.data_ptr()
    assert work.get_future() is handle.future


def test_wait_rethrows_cached_transport_error_without_restarting(extension) -> None:
    handle = _Handle(error=RuntimeError("transport failed"))
    work = extension.CompressedWork(object(), handle)

    with pytest.raises(RuntimeError, match="transport failed"):
        work.wait()
    with pytest.raises(RuntimeError, match="transport failed"):
        work.wait()

    assert handle.wait_calls == 1


def test_resources_are_retained_until_work_reaches_terminal_state(extension) -> None:
    class Resource:
        pass

    resource = Resource()
    resource_ref = weakref.ref(resource)
    work = extension.CompressedWork(
        object(), _Handle(), None, [resource]
    )
    del resource
    gc.collect()
    assert resource_ref() is not None
    assert len(work.resources) == 1

    work.wait()
    gc.collect()
    assert resource_ref() is None
    assert work.resources == ()


def test_native_work_waits_for_a_real_cuda_event(cuda_extension) -> None:
    import torch

    stream = torch.cuda.Stream()
    result = torch.zeros(1024, device="cuda")
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        result.fill_(7)
        event.record(stream)

    work = cuda_extension.CompressedWork(result, None, event, [result])

    assert work.uses_native_completion is True
    assert isinstance(work.query(), bool)
    assert work.wait().sum().item() == 7 * result.numel()


def test_native_work_releases_workspace_under_repeated_lifecycle_pressure(cuda_extension) -> None:
    import torch

    references = []
    for _ in range(128):
        resource = torch.ones(32, device="cuda")
        references.append(weakref.ref(resource))
        work = cuda_extension.CompressedWork(
            torch.ones(1, device="cuda"), _Handle(), None, [resource]
        )
        del resource
        assert references[-1]() is not None
        work.wait()
        del work

    gc.collect()
    assert all(reference() is None for reference in references)


def test_wait_completion_prefers_stream_wait_over_host_synchronize(extension) -> None:
    calls = []

    class Completion:
        def wait(self):
            calls.append("wait")

        def synchronize(self):
            calls.append("synchronize")

    work = extension.CompressedWork(object(), None, Completion())

    work.wait()

    assert calls == ["wait"]


def test_concurrent_wait_runs_failing_callback_at_most_once(extension) -> None:
    callback_calls = 0
    callback_lock = threading.Lock()
    start = threading.Barrier(3)
    errors = []

    class SlowHandle:
        def wait(self):
            time.sleep(0.05)

    def fail():
        nonlocal callback_calls
        with callback_lock:
            callback_calls += 1
        raise RuntimeError("callback failed")

    work = extension.CompressedWork(object(), SlowHandle(), None, [], fail)

    def wait_in_thread():
        start.wait()
        try:
            work.wait()
        except RuntimeError as error:
            errors.append(str(error))

    threads = [threading.Thread(target=wait_in_thread) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert callback_calls == 1
    assert errors == ["callback failed", "callback failed"]
