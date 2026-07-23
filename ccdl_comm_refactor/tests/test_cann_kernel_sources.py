from pathlib import Path


def test_cann_pybind_exports_linear_int8_symbols() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pybind = project_root / "ccdl_comm" / "csrc_ascend" / "pybind.cpp"
    source = pybind.read_text(encoding="utf-8")

    assert 'm.def("quantize_linear_int8"' in source
    assert 'm.def("dequantize_linear_int8"' in source


def test_cann_host_source_contains_real_linear_int8_logic() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / "ccdl_comm" / "csrc_ascend" / "quant_linear_int8.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "not implemented yet" not in source
    assert "amax" in source
    assert "round" in source
    assert "clamp" in source
