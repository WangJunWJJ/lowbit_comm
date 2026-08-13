"""Immutable Semantic IR root."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ErrorFeedbackDomain


@dataclass(frozen=True, slots=True)
class CommunicationProgram:
    operation: object
    output: object
    wire: object
    algorithm: object
    async_op: bool = True
    error_feedback: ErrorFeedbackDomain = ErrorFeedbackDomain.NONE

    def __post_init__(self) -> None:
        for name in ("operation", "output", "wire", "algorithm"):
            if getattr(self, name) is None:
                raise TypeError(f"{name} must not be None")
        if not isinstance(self.async_op, bool):
            raise TypeError("async_op must be a bool")
        if not isinstance(self.error_feedback, ErrorFeedbackDomain):
            raise TypeError("error_feedback must be an ErrorFeedbackDomain")
