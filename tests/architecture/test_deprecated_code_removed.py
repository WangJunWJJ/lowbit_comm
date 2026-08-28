"""Architecture guards for removed compatibility-only code."""

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

import lowbit_comm.compiler.evidence as evidence
from lowbit_comm.api.policy import (
    AccumulationDType,
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.signatures import strategy_signature


def test_compiler_evidence_exposes_only_the_current_schema() -> None:
    assert evidence.EVIDENCE_SCHEMA_VERSION == 3
    assert not hasattr(evidence, "LEGACY_EVIDENCE_SCHEMA_VERSION")
    assert not hasattr(evidence, "LegacyEvidenceMetrics")
    assert not hasattr(evidence, "LegacyEvidenceRecord")

    for removed_schema in (1, 2):
        with pytest.raises(CompileError, match="schema version must be 3"):
            evidence.EvidenceKey.from_mapping(
                schema_version=removed_schema,
                dimensions={},
            )


def test_strategy_signature_covers_every_strategy_field() -> None:
    baseline = StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_REDUCE_SCATTER,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=64,
        error_feedback=True,
        overlap=False,
    )
    variants = (
        baseline,
        StrategySpec(
            compression=CompressionKind.NONE,
            collective=CollectiveKind.NATIVE,
            topology=baseline.topology,
            error_feedback=baseline.error_feedback,
        ),
        replace(
            baseline,
            collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        ),
        replace(baseline, topology=TopologyKind.RING),
        replace(baseline, group_size=32),
        replace(baseline, accumulation_dtype=AccumulationDType.FP16),
        replace(baseline, error_feedback=False),
        replace(baseline, parameter_error_feedback=True),
        replace(baseline, overlap=True),
        replace(baseline, workspace_budget_bytes=1),
    )

    signatures = tuple(strategy_signature(strategy) for strategy in variants)

    assert len(set(signatures)) == len(variants)
    assert [component[0] for component in json.loads(signatures[0])] == [
        field.name for field in fields(StrategySpec)
    ]


def test_psi_training_state_compatibility_module_is_deleted() -> None:
    repository = Path(__file__).parents[2]

    assert not (repository / "tests/benchmarks/psi_v040_state.py").exists()
