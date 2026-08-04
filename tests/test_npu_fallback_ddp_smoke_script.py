from pathlib import Path


def test_npu_fallback_ddp_smoke_uses_payload_aware_all_gather() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tests" / "distributed" / "npu_torch_fallback_ddp_smoke.py"
    source = script.read_text(encoding="utf-8")

    assert "backend=\"hccl\"" in source
    assert "quantize_tensor_fallback" in source
    assert "dequantize_tensor_fallback" in source
    assert "_make_payload_all_gather(make_torch_all_gather())" in source
