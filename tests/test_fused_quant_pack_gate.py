from __future__ import annotations

from tests.benchmarks.fused_quant_pack_gate import evaluate


def test_residual_quant_pack_gate_rejects_task10_regression() -> None:
    failures = evaluate(candidate_ms=0.17072, baseline_ms=0.15477)

    assert failures == ["residual quant-pack regression: 0.170720 > 0.154770 ms"]


def test_residual_quant_pack_gate_accepts_equal_or_faster_candidate() -> None:
    assert evaluate(candidate_ms=0.15477, baseline_ms=0.15477) == []
    assert evaluate(candidate_ms=0.14703, baseline_ms=0.15477) == []
