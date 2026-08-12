from __future__ import annotations

import pytest

from lowbit_comm.runtime import CompletionWork, ManualCompletionEvent


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
