"""Tests for generic communication completion contracts."""

from typing import cast

import pytest

from lowbit_comm.core.errors import ExecutionError
from lowbit_comm.runtime.work import (
    CommunicationWork,
    CompletedWork,
    FailedWork,
)


def _consume_work(work: CommunicationWork[int]) -> int:
    """Exercise the structural protocol from a typed consumer."""
    if not work.is_completed():
        return work.wait()
    return work.result()


def test_completed_work_publishes_result() -> None:
    work = CompletedWork(17)

    assert work.is_completed() is True
    assert work.wait() == 17
    assert work.result() == 17
    assert _consume_work(work) == 17


def test_completed_work_preserves_result_identity() -> None:
    result = object()
    work = CompletedWork(result)

    assert work.wait() is result
    assert work.result() is result


def test_failed_work_never_publishes_result() -> None:
    failure = ExecutionError("transport failed")
    work = FailedWork[int](failure)

    assert work.is_completed() is True
    for operation in (work.wait, work.result, work.wait, work.result):
        with pytest.raises(ExecutionError, match="transport failed") as caught:
            operation()
        assert caught.value is failure


def test_failed_work_rejects_unsafe_exception_types() -> None:
    invalid_failure = cast(ExecutionError, ValueError("wrong boundary"))

    with pytest.raises(
        TypeError,
        match="FailedWork failure must be an ExecutionError",
    ):
        FailedWork[int](invalid_failure)
