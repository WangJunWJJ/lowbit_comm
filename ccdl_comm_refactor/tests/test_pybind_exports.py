from pathlib import Path


def test_pybind_exports_profileable_inplace_quantization_api() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pybind = project_root / "ccdl_comm" / "csrc" / "pybind.cpp"
    source = pybind.read_text(encoding="utf-8")

    assert 'm.def("quantize", &quantize);' in source
    assert 'm.def("dequantize", &dequantize);' in source
    assert 'm.def("inplace_quantize", &inplace_quantize);' in source
    assert 'm.def("inplace_dequantize", &inplace_dequantize);' in source
    assert 'm.def("dequantize_reduce", &dequantize_reduce);' in source
    assert 'm.def("inplace_dequantize_reduce", &inplace_dequantize_reduce);' in source
