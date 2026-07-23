from pathlib import Path


def test_cann_collective_perf_script_reports_cann_metrics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tests" / "distributed" / "cann_collective_perf.py"
    source = script.read_text(encoding="utf-8")

    assert "quantize_tensor_cann" in source
    assert "dequantize_tensor_cann" in source
    assert '"ccdl_cann_ms"' in source
    assert '"relative_l2"' in source
    assert "detect_cann" in source
    assert '"cann_diagnostics"' in source


def test_npu_cann_ddp_smoke_uses_cann_codec() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tests" / "distributed" / "npu_cann_ddp_smoke.py"
    source = script.read_text(encoding="utf-8")

    assert "quantize_tensor_cann" in source
    assert "dequantize_tensor_cann" in source
    assert "backend=\"hccl\"" in source
