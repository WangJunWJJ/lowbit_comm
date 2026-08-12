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


def test_dequant_reduce_tries_fused_kernel_before_fallback() -> None:
    source_root = Path(__file__).resolve().parents[1] / "ccdl_comm" / "csrc"
    pybind_source = (source_root / "pybind.cpp").read_text(encoding="utf-8")
    header_source = (source_root / "quantization" / "dequant_api.cuh").read_text(encoding="utf-8")
    kernel_source = (source_root / "quantization" / "dequant_reduce_kernel.cu").read_text(encoding="utf-8")

    assert "try_inplace_dequantize_reduce_fused(inputs, output, group_size, topk, bit, quant_type, compact)" in pybind_source
    assert "bool try_inplace_dequantize_reduce_fused" in header_source
    assert "dequant_reduce_fused_16bit_kernel" in kernel_source
    assert "dequant_reduce_fused_fp32_kernel" in kernel_source


def test_pybind_exports_native_error_feedback_update_kernel() -> None:
    source_root = Path(__file__).resolve().parents[1] / "ccdl_comm" / "csrc"
    pybind_source = (source_root / "pybind.cpp").read_text(encoding="utf-8")
    header_source = (source_root / "quantization" / "dequant_api.cuh").read_text(encoding="utf-8")
    kernel_source = (source_root / "quantization" / "dequant_reduce_kernel.cu").read_text(encoding="utf-8")

    assert "void inplace_error_feedback_update" in header_source
    assert "error_feedback_update_kernel" in kernel_source
    assert 'm.def("inplace_error_feedback_update", &inplace_error_feedback_update);' in pybind_source


def test_pybind_exports_combined_dequant_reduce_error_feedback_update() -> None:
    source_root = Path(__file__).resolve().parents[1] / "ccdl_comm" / "csrc"
    pybind_source = (source_root / "pybind.cpp").read_text(encoding="utf-8")

    assert "dequantize_reduce_update_error_feedback" in pybind_source
    assert "inplace_error_feedback_update(prepared, restored, residual)" in pybind_source
    assert 'm.def("dequantize_reduce_update_error_feedback", &dequantize_reduce_update_error_feedback);' in pybind_source


def test_pybind_exports_inplace_fused_dequant_reduce_mean_feedback_update() -> None:
    source_root = Path(__file__).resolve().parents[1] / "ccdl_comm" / "csrc"
    pybind_source = (source_root / "pybind.cpp").read_text(encoding="utf-8")
    header_source = (source_root / "quantization" / "dequant_api.cuh").read_text(encoding="utf-8")
    kernel_source = (source_root / "quantization" / "dequant_reduce_kernel.cu").read_text(encoding="utf-8")

    assert "bool inplace_dequantize_reduce_mean_update_error_feedback" in header_source
    assert "dequant_reduce_mean_feedback_fused_16bit_kernel" in kernel_source
    assert "dequant_reduce_mean_feedback_fused_fp32_kernel" in kernel_source
    assert (
        'm.def("inplace_dequantize_reduce_mean_update_error_feedback", '
        "&inplace_dequantize_reduce_mean_update_error_feedback);"
    ) in pybind_source
    assert "input.numel() != expected_input_numel" in kernel_source
    assert "bool inplace_dequantize_reduce_update_local_error_feedback" in header_source
    assert (
        'm.def("inplace_dequantize_reduce_update_local_error_feedback", '
        "&inplace_dequantize_reduce_update_local_error_feedback);"
    ) in pybind_source
    assert "prepared[index]) - local_restored" in kernel_source


def test_pybind_exports_native_cuda_work() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pybind_source = (project_root / "ccdl_comm" / "csrc" / "pybind.cpp").read_text(
        encoding="utf-8"
    )

    assert "bind_compressed_work(m);" in pybind_source
    assert "bind_cuda_executor(m);" in pybind_source
