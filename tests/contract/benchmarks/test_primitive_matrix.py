from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_primitive_matrix_reports_effective_paths_and_latency_quantiles() -> None:
    source = (ROOT / "tests/contract/benchmarks/primitive_matrix.py").read_text(
        encoding="utf-8"
    )

    for mode in (
        '"native_all_reduce"',
        '"compressed_all_gather"',
        '"compressed_reduce_scatter"',
        '"compressed_rs_ag"',
        '"hierarchical_compressed"',
    ):
        assert mode in source
    for field in (
        '"physical_primitive"',
        '"output"',
        '"p50_ms"',
        '"p95_ms"',
        '"peak_memory_bytes"',
    ):
        assert field in source
    assert "GroupedTransportRuntime" in source
    assert 'LOWBIT_COMM_TOPOLOGY' in source
