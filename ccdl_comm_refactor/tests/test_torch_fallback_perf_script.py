from pathlib import Path


def test_torch_fallback_perf_script_supports_hccl_and_reports_error_metrics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tests" / "distributed" / "torch_fallback_collective_perf.py"
    source = script.read_text(encoding="utf-8")

    assert "hccl" in source
    assert "quantize_tensor_fallback" in source
    assert '"torch_all_reduce_ms"' in source
    assert '"ccdl_torch_fallback_ms"' in source
    assert '"relative_l2"' in source
