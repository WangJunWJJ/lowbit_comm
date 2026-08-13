"""Evidence-gated compile-time strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

from lowbit_comm.core.context import CompileContext


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    evidence_id: str
    target: str
    topology_signature: str
    world_size: int
    speedup_percent: float

    def matches(self, target: str, context: CompileContext) -> bool:
        return (
            self.target == target
            and self.topology_signature == context.topology_signature
            and self.world_size == context.world_size
        )


@dataclass(frozen=True, slots=True)
class AutoDecision:
    use_compression: bool
    fallback_reason: str | None
    evidence_id: str | None


def decide_auto(
    target: str,
    context: CompileContext,
    evidence: BenchmarkEvidence | None,
) -> AutoDecision:
    if evidence is None:
        return AutoDecision(False, "missing benchmark evidence", None)
    if not evidence.matches(target, context):
        return AutoDecision(False, "benchmark evidence mismatch", None)
    if evidence.speedup_percent <= 0:
        return AutoDecision(False, "benchmark evidence shows no speedup", evidence.evidence_id)
    return AutoDecision(True, None, evidence.evidence_id)
