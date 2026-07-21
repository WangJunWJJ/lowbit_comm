from pathlib import Path


def test_pybind_exports_only_facade_required_quantization_api() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pybind = project_root / "ccdl_comm" / "csrc" / "pybind.cpp"
    source = pybind.read_text(encoding="utf-8")

    assert 'm.def("quantize", &quantize);' in source
    assert 'm.def("dequantize", &dequantize);' in source
    assert "inplace_quantize" not in source
    assert "inplace_dequantize" not in source
