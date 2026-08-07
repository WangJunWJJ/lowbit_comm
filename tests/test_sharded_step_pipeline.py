from __future__ import annotations

import pytest

from ccdl_comm.communication.sharded_step import ShardedStepPipeline


class FakeConsumer:
    def __init__(self, bucket_id: str, calls: list[object]) -> None:
        self._bucket_id = bucket_id
        self._calls = calls

    def consume(self, reduced, *, step: int):
        self._calls.append(("consume", self._bucket_id, reduced, step))
        return f"updated-{self._bucket_id}"


class FakeWork:
    def __init__(
        self,
        bucket_id: str,
        result: object,
        waits: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.bucket_id = bucket_id
        self.result = result
        self.waits = waits
        self.error = error
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1
        self.waits.append(self.bucket_id)
        if self.error is not None:
            raise self.error
        return self.result


class FakeRestore:
    def __init__(
        self,
        bucket_id: str,
        calls: list[object],
        waits: list[str],
        works: dict[str, FakeWork],
        errors: dict[str, BaseException],
    ) -> None:
        self._bucket_id = bucket_id
        self._calls = calls
        self._waits = waits
        self._works = works
        self._errors = errors

    def restore(self, updated, *, out, async_op: bool):
        self._calls.append(("restore", self._bucket_id, updated, out, async_op))
        work = FakeWork(
            self._bucket_id,
            out,
            self._waits,
            error=self._errors.get(self._bucket_id),
        )
        self._works[self._bucket_id] = work
        return work


def make_pipeline(*, max_inflight: int = 2, errors=None):
    calls: list[object] = []
    waits: list[str] = []
    works: dict[str, FakeWork] = {}
    failures = errors or {}
    consumers = {
        f"bucket{index}": FakeConsumer(f"bucket{index}", calls)
        for index in range(4)
    }
    restores = {
        bucket_id: FakeRestore(bucket_id, calls, waits, works, failures)
        for bucket_id in consumers
    }
    pipeline = ShardedStepPipeline(
        consumer_for_bucket=consumers.__getitem__,
        restore_for_bucket=restores.__getitem__,
        max_inflight=max_inflight,
    )
    return pipeline, calls, waits, works


def test_pipeline_waits_oldest_work_at_inflight_limit() -> None:
    pipeline, _calls, _waits, works = make_pipeline(max_inflight=2)

    first = pipeline.consume_bucket(
        "bucket0", "reduced0", parameter_view="view0", step=1
    )
    second = pipeline.consume_bucket(
        "bucket1", "reduced1", parameter_view="view1", step=1
    )
    third = pipeline.consume_bucket(
        "bucket2", "reduced2", parameter_view="view2", step=1
    )

    assert first is works["bucket0"]
    assert first.wait_count == 1
    assert second.wait_count == 0
    assert third.wait_count == 0


def test_finish_step_waits_every_bucket_in_submission_order() -> None:
    pipeline, _calls, waits, _works = make_pipeline(max_inflight=4)
    for index in range(3):
        pipeline.consume_bucket(
            f"bucket{index}",
            f"reduced{index}",
            parameter_view=f"view{index}",
            step=1,
        )

    result = pipeline.finish_step()

    assert waits == ["bucket0", "bucket1", "bucket2"]
    assert result == ("view0", "view1", "view2")


def test_consumer_update_precedes_restore_for_each_bucket() -> None:
    pipeline, calls, _waits, _works = make_pipeline()

    pipeline.consume_bucket("bucket0", "gradient", parameter_view="view", step=1)

    assert calls == [
        ("consume", "bucket0", "gradient", 1),
        ("restore", "bucket0", "updated-bucket0", "view", True),
    ]


def test_second_step_cannot_start_before_finish_and_steps_are_monotonic() -> None:
    pipeline, _calls, _waits, _works = make_pipeline()
    pipeline.consume_bucket("bucket0", "reduced", parameter_view="view", step=1)

    with pytest.raises(RuntimeError, match="finish_step"):
        pipeline.consume_bucket("bucket1", "reduced", parameter_view="view", step=2)

    pipeline.finish_step()
    pipeline.consume_bucket("bucket0", "reduced", parameter_view="view", step=2)
    pipeline.finish_step()

    with pytest.raises(ValueError, match="monotonically increasing"):
        pipeline.consume_bucket("bucket0", "reduced", parameter_view="view", step=2)


def test_failed_work_is_retained_and_reraised_by_finish_step() -> None:
    error = RuntimeError("restore failed")
    pipeline, _calls, waits, works = make_pipeline(errors={"bucket0": error})
    pipeline.consume_bucket("bucket0", "reduced0", parameter_view="view0", step=1)
    pipeline.consume_bucket("bucket1", "reduced1", parameter_view="view1", step=1)

    with pytest.raises(RuntimeError, match="restore failed"):
        pipeline.finish_step()

    assert waits == ["bucket0"]
    assert works["bucket1"].wait_count == 0
    with pytest.raises(RuntimeError, match="finish_step"):
        pipeline.consume_bucket("bucket2", "reduced2", parameter_view="view2", step=2)


@pytest.mark.parametrize("max_inflight", (True, 0, -1, 1.5))
def test_invalid_max_inflight_is_rejected(max_inflight: object) -> None:
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        ShardedStepPipeline(
            consumer_for_bucket=lambda bucket_id: bucket_id,
            restore_for_bucket=lambda bucket_id: bucket_id,
            max_inflight=max_inflight,
        )
