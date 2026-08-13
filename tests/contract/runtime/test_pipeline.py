from __future__ import annotations

from concurrent.futures import Future

import pytest

from lowbit_comm.runtime import (
    CompletionManager,
    CompletionPipeline,
    ManualCompletionEvent,
)


class Lease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def test_pipeline_future_waits_for_transport_postprocess_and_output() -> None:
    transport = ManualCompletionEvent()
    output = ManualCompletionEvent()
    lease = Lease()
    calls: list[str] = []
    pipeline = CompletionPipeline("pending", resources=(lease,))
    pipeline.add_stage(
        "transport",
        transport,
        action=lambda result: calls.append("postprocess") or "ready",
    )
    pipeline.add_stage("output", output)
    future = pipeline.get_future(Future)

    assert pipeline.query() is False
    assert calls == []

    transport.complete()
    assert pipeline.query() is False
    assert calls == ["postprocess"]
    assert future.done() is False
    assert lease.released is False

    output.complete()
    assert pipeline.query() is True
    assert future.result() == "ready"
    assert lease.released is True


def test_pipeline_failure_is_stable_and_releases_resources_once() -> None:
    event = ManualCompletionEvent(completed=True)
    lease = Lease()
    pipeline = CompletionPipeline("pending", resources=(lease,))

    def fail(result: str) -> str:
        raise ValueError("postprocess failed")

    pipeline.add_stage("postprocess", event, action=fail)

    with pytest.raises(ValueError, match="postprocess failed"):
        pipeline.wait()
    with pytest.raises(ValueError, match="postprocess failed"):
        pipeline.wait()
    assert lease.released is True


def test_pipeline_rejects_stage_mutation_after_submission() -> None:
    pipeline = CompletionPipeline("pending")
    pipeline.add_stage("transport", ManualCompletionEvent(completed=True))
    pipeline.get_future(Future)

    with pytest.raises(RuntimeError, match="submitted"):
        pipeline.add_stage("late", ManualCompletionEvent(completed=True))


def test_completion_manager_resolves_future_without_caller_polling() -> None:
    first_event = ManualCompletionEvent()
    second_event = ManualCompletionEvent()
    manager = CompletionManager(poll_interval=0.001)
    first = CompletionPipeline("first")
    second = CompletionPipeline("second")
    first.add_stage("first", first_event)
    second.add_stage("second", second_event)
    first_future = first.get_future(Future, manager=manager)
    second_future = second.get_future(Future, manager=manager)

    second_event.complete()
    assert second_future.result(timeout=1.0) == "second"
    assert first_future.done() is False

    first_event.complete()
    assert first_future.result(timeout=1.0) == "first"
    manager.close()
