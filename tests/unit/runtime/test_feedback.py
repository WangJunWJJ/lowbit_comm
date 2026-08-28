"""Tests for transactional error-feedback publication."""

import inspect
from typing import cast

import pytest

from lowbit_comm.core.errors import ExecutionError
from lowbit_comm.runtime.feedback import (
    ErrorFeedbackTransaction,
    FeedbackState,
)


def test_feedback_commits_only_after_successful_prepare() -> None:
    transaction = ErrorFeedbackTransaction(
        previous=(1.0,),
        candidate=(0.25,),
    )

    transaction.prepare()

    assert transaction.commit() == (0.25,)
    assert transaction.state is FeedbackState.COMMITTED


def test_phase_one_prepare_and_commit_require_no_completion_token() -> None:
    assert tuple(inspect.signature(ErrorFeedbackTransaction.prepare).parameters) == (
        "self",
    )
    assert tuple(inspect.signature(ErrorFeedbackTransaction.commit).parameters) == (
        "self",
    )

    candidate = object()
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=candidate,
    )
    transaction.prepare()

    assert transaction.commit() is candidate


def test_candidate_is_hidden_until_commit() -> None:
    previous = object()
    candidate = object()
    transaction = ErrorFeedbackTransaction(
        previous=previous,
        candidate=candidate,
    )

    assert transaction.state is FeedbackState.CREATED
    assert transaction.visible_residual is previous
    assert not hasattr(transaction, "candidate")

    transaction.prepare()

    assert transaction.state is FeedbackState.PREPARED
    assert transaction.visible_residual is previous

    assert transaction.commit() is candidate
    assert transaction.visible_residual is candidate


@pytest.mark.parametrize("should_prepare", [False, True])
def test_abort_preserves_previous_residual(
    should_prepare: bool,
) -> None:
    previous = object()
    transaction = ErrorFeedbackTransaction(
        previous=previous,
        candidate=object(),
    )
    if should_prepare:
        transaction.prepare()

    transaction.abort(ExecutionError("kernel failed"))

    assert transaction.state is FeedbackState.ABORTED
    assert transaction.visible_residual is previous


def test_aborted_feedback_cannot_commit() -> None:
    failure = ExecutionError("kernel failed")
    transaction = ErrorFeedbackTransaction(
        previous=(1.0,),
        candidate=(0.25,),
    )
    transaction.prepare()
    transaction.abort(failure)

    with pytest.raises(ExecutionError, match="aborted") as caught:
        transaction.commit()

    assert caught.value.__cause__ is failure
    assert transaction.visible_residual == (1.0,)


def test_prepare_after_abort_preserves_original_failure() -> None:
    failure = ExecutionError("transport failed")
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )
    transaction.abort(failure)

    with pytest.raises(ExecutionError, match="aborted") as caught:
        transaction.prepare()

    assert caught.value.__cause__ is failure


def test_commit_before_prepare_is_rejected() -> None:
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )

    with pytest.raises(ExecutionError, match="before prepare"):
        transaction.commit()

    assert transaction.state is FeedbackState.CREATED


def test_double_prepare_is_rejected() -> None:
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )
    transaction.prepare()

    with pytest.raises(ExecutionError, match="already prepared"):
        transaction.prepare()

    assert transaction.state is FeedbackState.PREPARED


def test_double_commit_is_rejected() -> None:
    candidate = object()
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=candidate,
    )
    transaction.prepare()
    assert transaction.commit() is candidate

    with pytest.raises(ExecutionError, match="already committed"):
        transaction.commit()

    assert transaction.state is FeedbackState.COMMITTED
    assert transaction.visible_residual is candidate


def test_double_abort_is_rejected_without_replacing_reason() -> None:
    original = ExecutionError("transport failed")
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )
    transaction.abort(original)

    with pytest.raises(ExecutionError, match="already aborted") as caught:
        transaction.abort(ExecutionError("replacement"))

    assert caught.value.__cause__ is original
    assert transaction.state is FeedbackState.ABORTED


@pytest.mark.parametrize(
    "terminal_state",
    [FeedbackState.COMMITTED, FeedbackState.ABORTED],
)
def test_prepare_after_terminal_state_is_rejected(
    terminal_state: FeedbackState,
) -> None:
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )
    transaction.prepare()
    if terminal_state is FeedbackState.COMMITTED:
        transaction.commit()
    else:
        transaction.abort(ExecutionError("failed"))

    with pytest.raises(ExecutionError, match=terminal_state.value):
        transaction.prepare()

    assert transaction.state is terminal_state


def test_abort_after_commit_is_rejected() -> None:
    candidate = object()
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=candidate,
    )
    transaction.prepare()
    transaction.commit()

    with pytest.raises(ExecutionError, match="committed"):
        transaction.abort(ExecutionError("too late"))

    assert transaction.state is FeedbackState.COMMITTED
    assert transaction.visible_residual is candidate


@pytest.mark.parametrize(
    "reason",
    [ValueError("wrong type"), "kernel failed", 1, True, None],
)
def test_abort_rejects_non_execution_error_reason(reason: object) -> None:
    transaction = ErrorFeedbackTransaction(
        previous=object(),
        candidate=object(),
    )

    with pytest.raises(
        TypeError,
        match="Abort reason must be an ExecutionError",
    ):
        transaction.abort(cast(ExecutionError, reason))

    assert transaction.state is FeedbackState.CREATED
