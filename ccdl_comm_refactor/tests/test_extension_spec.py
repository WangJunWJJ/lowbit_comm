from pathlib import Path

from ccdl_comm.build.extension import collect_cuda_sources, create_cuda_extension


def test_collect_cuda_sources_includes_generated_sources_and_binding(tmp_path):
    (tmp_path / "pybind.cpp").write_text("// binding\n", encoding="utf-8")
    quantization = tmp_path / "quantization"
    quantization.mkdir()
    for name in ["utils.cu", "gen_quant_api.cu", "gen_dequant_api.cu"]:
        (quantization / name).write_text("// source\n", encoding="utf-8")

    sources = collect_cuda_sources(tmp_path)

    assert sources == (
        tmp_path / "pybind.cpp",
        quantization / "gen_dequant_api.cu",
        quantization / "gen_quant_api.cu",
        quantization / "utils.cu",
    )


def test_create_cuda_extension_ensures_generated_sources_before_factory_call(tmp_path):
    (tmp_path / "pybind.cpp").write_text("// binding\n", encoding="utf-8")
    quantization = tmp_path / "quantization"
    quantization.mkdir()
    (quantization / "utils.cu").write_text("// source\n", encoding="utf-8")

    calls = []

    def run_generator(command):
        calls.append(command)
        script_name = Path(command[1]).name
        if script_name == "gen_code_quant.py":
            (quantization / "gen_quant_api.cu").write_text("torch::Tensor quantize(", encoding="utf-8")
        elif script_name == "gen_code_dequant.py":
            (quantization / "gen_dequant_api.cu").write_text("torch::Tensor dequantize(", encoding="utf-8")

    def extension_factory(**kwargs):
        return kwargs

    extension = create_cuda_extension(
        tmp_path,
        run_generator=run_generator,
        extension_factory=extension_factory,
    )

    assert len(calls) == 2
    assert extension["name"] == "ccdl_cuda_ops"
    assert extension["sources"] == [
        str(tmp_path / "pybind.cpp"),
        str(quantization / "gen_dequant_api.cu"),
        str(quantization / "gen_quant_api.cu"),
        str(quantization / "utils.cu"),
    ]
    assert extension["extra_compile_args"]["nvcc"] == ["-O3", "-U__CUDA_NO_HALF_OPERATORS__"]
