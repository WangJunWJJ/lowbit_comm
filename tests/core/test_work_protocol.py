from __future__ import annotations

from ccdl_comm.collectives.work import (
    CollectiveWork as LegacyCollectiveWork,
    CompletionWork as LegacyCompletionWork,
    ImmediateWork as LegacyImmediateWork,
)
from ccdl_comm.work import CollectiveWork, CompletionWork, ImmediateWork


def test_legacy_work_module_reexports_core_implementations() -> None:
    assert LegacyCollectiveWork is CollectiveWork
    assert LegacyCompletionWork is CompletionWork
    assert LegacyImmediateWork is ImmediateWork


def test_immediate_work_returns_completed_result() -> None:
    work = ImmediateWork("done")

    assert work.query() is True
    assert work.wait() == "done"


class ControlledFuture:
    def __init__(self) -> None:
        self._callbacks = []
        self._done = False
        self._result = None
        self._exception = None

    def then(self, callback):
        self._callbacks.append(callback)
        return self

    def finish_transport(self) -> None:
        for callback in tuple(self._callbacks):
            callback(self)

    def set_result(self, result) -> None:
        self._result = result
        self._done = True

    def set_exception(self, exception) -> None:
        self._exception = exception
        self._done = True

    def done(self) -> bool:
        return self._done

    def wait(self):
        if self._exception is not None:
            raise self._exception
        return self._result


class ControlledHandle:
    def __init__(self, future: ControlledFuture) -> None:
        self.future = future
        self.wait_calls = 0

    def get_future(self):
        return self.future

    def wait(self) -> None:
        self.wait_calls += 1


def test_completion_work_future_represents_deferred_completion() -> None:
    inner = ControlledFuture()
    handle = ControlledHandle(inner)
    callbacks = []
    work = CompletionWork(
        None,
        handle=handle,
        complete=lambda: callbacks.append("decoded") or "decoded",
        future_factory=ControlledFuture,
    )

    outer = work.get_future()
    assert outer is not inner
    assert not outer.done()

    inner.finish_transport()

    assert outer.done()
    assert outer.wait() == "decoded"
    assert work.wait() == "decoded"
    assert callbacks == ["decoded"]
    assert handle.wait_calls == 1


def test_completion_work_does_not_expose_transport_future_without_outer_factory() -> None:
    inner = ControlledFuture()
    work = CompletionWork(None, handle=ControlledHandle(inner), complete=lambda: "decoded")

    assert work.get_future() is None


def test_completion_work_propagates_callback_error_to_wait_and_outer_future() -> None:
    inner = ControlledFuture()
    error = RuntimeError("decode failed")
    callbacks = []

    def fail_decode():
        callbacks.append("decode")
        raise error

    work = CompletionWork(
        None,
        handle=ControlledHandle(inner),
        complete=fail_decode,
        future_factory=ControlledFuture,
    )
    outer = work.get_future()

    inner.finish_transport()

    assert outer.done()
    try:
        outer.wait()
    except RuntimeError as exc:
        assert exc is error
    else:
        raise AssertionError("outer future did not preserve callback error")
    try:
        work.wait()
    except RuntimeError as exc:
        assert exc is error
    else:
        raise AssertionError("work did not preserve callback error")
    assert callbacks == ["decode"]
