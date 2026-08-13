from __future__ import annotations

import pytest
import gc
import weakref

from lowbit_comm.runtime import (
    CompletionOutcome,
    CompletionWork,
    ManualCompletionEvent,
)


def test_work_waits_for_event_before_running_completion() -> None:
    event = ManualCompletionEvent()
    calls: list[str] = []
    work = CompletionWork(
        "pending",
        event=event,
        complete=lambda: calls.append("complete") or "ready",
    )

    assert work.query() is False
    event.complete()
    assert work.wait() == "ready"
    assert calls == ["complete"]
    assert work.wait() == "ready"
    assert calls == ["complete"]


def test_work_preserves_completion_failure() -> None:
    event = ManualCompletionEvent(completed=True)

    def fail() -> str:
        raise ValueError("postprocess failed")

    work = CompletionWork("unused", event=event, complete=fail)
    with pytest.raises(ValueError, match="postprocess failed"):
        work.wait()
    with pytest.raises(ValueError, match="postprocess failed"):
        work.wait()


def test_work_retains_non_releasable_resources_until_completion() -> None:
    class Buffer:
        pass

    event = ManualCompletionEvent()
    buffer = Buffer()
    reference = weakref.ref(buffer)
    work = CompletionWork("ready", event=event, resources=(buffer,))
    del buffer
    gc.collect()

    assert reference() is not None
    event.complete()
    assert work.wait() == "ready"


def test_query_advances_ready_event_without_blocking() -> None:
    event = ManualCompletionEvent()
    calls: list[str] = []
    work = CompletionWork(
        "pending",
        event=event,
        complete=lambda: calls.append("complete") or "ready",
    )

    event.complete()

    assert work.query() is True
    assert calls == ["complete"]
    assert work.wait() == "ready"


def test_resources_wait_for_postprocess_output_event() -> None:
    class Lease:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    transport = ManualCompletionEvent(completed=True)
    output_ready = ManualCompletionEvent()
    lease = Lease()
    work = CompletionWork(
        "pending",
        event=transport,
        complete=lambda: CompletionOutcome("ready", output_ready),
        resources=(lease,),
    )

    assert work.query() is False
    assert lease.released is False
    output_ready.complete()
    assert work.query() is True
    assert lease.released is True
    assert work.wait() == "ready"


def test_wait_timeout_applies_to_output_ready_event() -> None:
    output_ready = ManualCompletionEvent()
    work = CompletionWork(
        "pending",
        event=ManualCompletionEvent(completed=True),
        complete=lambda: CompletionOutcome("ready", output_ready),
    )

    with pytest.raises(TimeoutError, match="communication work"):
        work.wait(timeout=0.001)
