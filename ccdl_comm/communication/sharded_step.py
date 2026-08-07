"""Bounded pipeline for reduced-shard updates and parameter restoration."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ccdl_comm.work import CollectiveWork


_PENDING = object()


@dataclass(slots=True)
class _Submission:
    bucket_id: Any
    work: CollectiveWork[Any]
    result: Any = _PENDING


class ShardedStepPipeline:
    """Overlap bucket restoration while bounding retained communication work."""

    def __init__(
        self,
        *,
        consumer_for_bucket: Callable[[Any], Any],
        restore_for_bucket: Callable[[Any], Any],
        max_inflight: int = 2,
    ) -> None:
        if not callable(consumer_for_bucket):
            raise TypeError("consumer_for_bucket must be callable")
        if not callable(restore_for_bucket):
            raise TypeError("restore_for_bucket must be callable")
        if (
            isinstance(max_inflight, bool)
            or not isinstance(max_inflight, int)
            or max_inflight < 1
        ):
            raise ValueError("max_inflight must be a positive integer")
        self._consumer_for_bucket = consumer_for_bucket
        self._restore_for_bucket = restore_for_bucket
        self._max_inflight = max_inflight
        self._active_step: int | None = None
        self._last_completed_step = 0
        self._submissions: list[_Submission] = []
        self._pending: deque[_Submission] = deque()
        self._bucket_ids: set[Any] = set()

    def consume_bucket(
        self,
        bucket_id: Any,
        reduced: Any,
        *,
        parameter_view: Any,
        step: int,
    ) -> CollectiveWork[Any]:
        """Update and submit one bucket, waiting the oldest at the depth limit."""

        _require_positive_step(step)
        started_step = self._begin_or_validate_step(step)
        if bucket_id in self._bucket_ids:
            raise ValueError("bucket_id must be submitted once per step")
        try:
            consumer = self._consumer_for_bucket(bucket_id)
            restore = self._restore_for_bucket(bucket_id)
            updated = consumer.consume(reduced, step=step)
            work = restore.restore(updated, out=parameter_view, async_op=True)
            if not callable(getattr(work, "wait", None)):
                raise TypeError("restore must return work exposing wait()")
        except BaseException:
            if started_step and not self._submissions:
                self._active_step = None
            raise

        submission = _Submission(bucket_id=bucket_id, work=work)
        self._submissions.append(submission)
        self._pending.append(submission)
        self._bucket_ids.add(bucket_id)
        if len(self._pending) > self._max_inflight:
            oldest = self._pending[0]
            self._wait_submission(oldest)
            self._pending.popleft()
        return work

    def finish_step(self) -> tuple[Any, ...]:
        """Wait all submitted buckets in order and close the active step."""

        if self._active_step is None:
            return ()
        for submission in self._submissions:
            self._wait_submission(submission)
        results = tuple(submission.result for submission in self._submissions)
        self._last_completed_step = self._active_step
        self._active_step = None
        self._submissions.clear()
        self._pending.clear()
        self._bucket_ids.clear()
        return results

    def _begin_or_validate_step(self, step: int) -> bool:
        if self._active_step is None:
            if step <= self._last_completed_step:
                raise ValueError("step must be monotonically increasing")
            self._active_step = step
            return True
        if step != self._active_step:
            raise RuntimeError("finish_step must complete before starting another step")
        return False

    @staticmethod
    def _wait_submission(submission: _Submission) -> None:
        if submission.result is _PENDING:
            submission.result = submission.work.wait()


def _require_positive_step(step: object) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")


__all__ = ["ShardedStepPipeline"]
