from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from ccdl_comm.config import CompressionConfig


@dataclass(frozen=True)
class ErrorFeedbackDecision:
    apply: bool
    update: bool
    reason: str


@dataclass
class ErrorFeedbackPolicy:
    config: CompressionConfig
    _steps: dict[Hashable, int] = field(default_factory=dict)

    def decide(self, key: Hashable, *, numel: int) -> ErrorFeedbackDecision:
        policy = self.config.effective_error_feedback_policy()
        step = self._steps.get(key, 0)
        if policy == "none":
            return ErrorFeedbackDecision(False, False, "error feedback disabled")
        if policy == "always":
            return ErrorFeedbackDecision(True, True, "error feedback policy always")
        if policy == "large_bucket_only":
            threshold = self.config.error_feedback_min_numel
            if numel < threshold:
                return ErrorFeedbackDecision(
                    False,
                    False,
                    f"bucket numel {numel} below error feedback threshold {threshold}",
                )
            return ErrorFeedbackDecision(
                True,
                True,
                f"bucket numel {numel} reached error feedback threshold {threshold}",
            )
        if policy == "warmup_then_enable":
            warmup = self.config.error_feedback_warmup_steps
            if step < warmup:
                return ErrorFeedbackDecision(False, False, f"bucket step {step} before error feedback warmup {warmup}")
            return ErrorFeedbackDecision(True, True, f"bucket step {step} reached error feedback warmup {warmup}")
        if policy == "periodic":
            period = self.config.error_feedback_period
            should_update = step % period == 0
            if should_update:
                return ErrorFeedbackDecision(
                    True,
                    True,
                    f"bucket step {step} updates error feedback every {period} steps",
                )
            return ErrorFeedbackDecision(
                True,
                False,
                f"bucket step {step} skips error feedback update until period {period}",
            )
        raise ValueError(f"unsupported error feedback policy: {policy}")

    def advance(self, key: Hashable) -> None:
        self._steps[key] = self._steps.get(key, 0) + 1
