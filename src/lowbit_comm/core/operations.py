"""Mathematical operations represented by the Semantic IR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReduceMean:
    """Compute the elementwise mean across participating ranks."""


@dataclass(frozen=True, slots=True)
class ReduceSum:
    """Compute the elementwise sum across participating ranks."""
