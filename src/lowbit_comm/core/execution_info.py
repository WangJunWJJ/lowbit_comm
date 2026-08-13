"""Immutable record of a compiler strategy decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    requested_algorithm: str
    effective_algorithm: str
    requested_wire: object
    effective_wire: object
    fallback_reason: str | None = None
    evidence_id: str | None = None
