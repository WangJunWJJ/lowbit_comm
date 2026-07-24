from pathlib import Path


def test_collective_perf_script_reports_compressed_all_gather_metrics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tests" / "distributed" / "collective_perf_compare.py"
    source = script.read_text(encoding="utf-8")

    assert "compressed_all_gather" in source
    assert '"torch_all_gather_ms"' in source
    assert '"ccdl_all_gather_ms"' in source
    assert '"all_gather_relative_l2"' in source
    assert "--compact" in source
    assert '"compact"' in source
