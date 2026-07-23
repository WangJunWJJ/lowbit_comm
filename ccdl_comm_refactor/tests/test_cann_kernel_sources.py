from pathlib import Path


def test_cann_pybind_exports_linear_int8_symbols() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pybind = project_root / "ccdl_comm" / "csrc_ascend" / "pybind.cpp"
    source = pybind.read_text(encoding="utf-8")

    assert 'm.def("quantize_linear_int8"' in source
    assert 'm.def("dequantize_linear_int8"' in source
