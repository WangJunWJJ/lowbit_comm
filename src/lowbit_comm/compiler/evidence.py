"""Immutable benchmark evidence catalog used only during compilation."""

from __future__ import annotations

from dataclasses import fields

from lowbit_comm.core import BackendCapabilities, CommunicationProgram, CompileContext

from .cost_model import BenchmarkEvidence


_IDENTITY_FIELDS = tuple(
    field.name
    for field in fields(BenchmarkEvidence)
    if field.name not in {"evidence_id", "speedup_percent"}
)


class EvidenceCatalog:
    def __init__(self, evidence: tuple[BenchmarkEvidence, ...]) -> None:
        self._evidence = tuple(evidence)
        if any(not isinstance(item, BenchmarkEvidence) for item in self._evidence):
            raise TypeError("catalog entries must be BenchmarkEvidence")
        identifiers = [item.evidence_id for item in self._evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate evidence_id in benchmark catalog")
        identities = [_identity(item) for item in self._evidence]
        if len(identities) != len(set(identities)):
            raise ValueError("conflicting benchmark evidence for one fingerprint")

    def match(
        self,
        target: str,
        context: CompileContext,
        program: CommunicationProgram,
        capabilities: BackendCapabilities,
    ) -> BenchmarkEvidence | None:
        matches = tuple(
            item
            for item in self._evidence
            if item.matches(target, context, program, capabilities)
        )
        if len(matches) > 1:
            raise RuntimeError("benchmark catalog produced ambiguous exact matches")
        return matches[0] if matches else None


def _identity(evidence: BenchmarkEvidence) -> tuple[object, ...]:
    return tuple(getattr(evidence, name) for name in _IDENTITY_FIELDS)
