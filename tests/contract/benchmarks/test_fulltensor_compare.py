from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_workspace_benchmark_reports_reuse_tail_latency_and_peak_memory() -> None:
    source = (ROOT / "tests/contract/benchmarks/fulltensor_compare.py").read_text(
        encoding="utf-8"
    )

    for field in (
        '"schema_version"',
        '"fingerprint"',
        '"samples_ms"',
        '"native_p50_ms"',
        '"native_p95_ms"',
        '"compressed_p50_ms"',
        '"compressed_p95_ms"',
        '"workspace_allocation_count"',
        '"workspace_reuse_count"',
        '"workspace_peak_in_use_bytes"',
        '"steady_state_new_allocations"',
        '"native_peak_memory_bytes"',
        '"compressed_peak_memory_bytes"',
    ):
        assert field in source
    assert "def _percentile(" in source
