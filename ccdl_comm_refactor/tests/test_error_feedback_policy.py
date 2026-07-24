from ccdl_comm.config import CompressionConfig
from ccdl_comm.quantization.error_feedback_policy import ErrorFeedbackPolicy


def test_none_policy_never_applies_or_updates() -> None:
    policy = ErrorFeedbackPolicy(CompressionConfig(error_feedback=False))

    decision = policy.decide("bucket0", numel=10_000)

    assert not decision.apply
    assert not decision.update
    assert decision.reason == "error feedback disabled"


def test_always_policy_applies_and_updates() -> None:
    policy = ErrorFeedbackPolicy(CompressionConfig(error_feedback=True, error_feedback_policy="always"))

    decision = policy.decide("bucket0", numel=10_000)

    assert decision.apply
    assert decision.update
    assert decision.reason == "error feedback policy always"


def test_large_bucket_only_uses_numel_threshold() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4096,
        )
    )

    small = policy.decide("bucket0", numel=1024)
    large = policy.decide("bucket1", numel=4096)

    assert not small.apply
    assert not small.update
    assert small.reason == "bucket numel 1024 below error feedback threshold 4096"
    assert large.apply
    assert large.update
    assert large.reason == "bucket numel 4096 reached error feedback threshold 4096"


def test_warmup_then_enable_uses_bucket_local_steps() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="warmup_then_enable",
            error_feedback_warmup_steps=2,
        )
    )

    first = policy.decide("bucket0", numel=10_000)
    policy.advance("bucket0")
    second = policy.decide("bucket0", numel=10_000)
    policy.advance("bucket0")
    third = policy.decide("bucket0", numel=10_000)

    assert not first.apply
    assert first.reason == "bucket step 0 before error feedback warmup 2"
    assert not second.apply
    assert second.reason == "bucket step 1 before error feedback warmup 2"
    assert third.apply
    assert third.update
    assert third.reason == "bucket step 2 reached error feedback warmup 2"


def test_periodic_policy_applies_every_step_but_updates_periodically() -> None:
    policy = ErrorFeedbackPolicy(
        CompressionConfig(
            error_feedback=True,
            error_feedback_policy="periodic",
            error_feedback_period=3,
        )
    )

    decisions = []
    for _ in range(4):
        decisions.append(policy.decide("bucket0", numel=10_000))
        policy.advance("bucket0")

    assert [decision.apply for decision in decisions] == [True, True, True, True]
    assert [decision.update for decision in decisions] == [True, False, False, True]
    assert decisions[0].reason == "bucket step 0 updates error feedback every 3 steps"
    assert decisions[1].reason == "bucket step 1 skips error feedback update until period 3"
