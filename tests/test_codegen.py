from pathlib import Path

import pytest

from ccdl_comm.build.codegen import GENERATED_SOURCE_NAMES, ensure_generated_sources, missing_generated_sources


CUDA_SOURCE_DIR = Path(__file__).parents[1] / "ccdl_comm" / "csrc" / "quantization"


def test_missing_generated_sources_reports_absent_generated_cuda_files(tmp_path):
    missing = missing_generated_sources(tmp_path)

    assert missing == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_missing_generated_sources_reports_empty_generated_cuda_files(tmp_path):
    for name in GENERATED_SOURCE_NAMES:
        (tmp_path / name).write_text("", encoding="utf-8")

    missing = missing_generated_sources(tmp_path)

    assert missing == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_missing_generated_sources_reports_stub_generated_cuda_files(tmp_path):
    (tmp_path / "gen_quant_api.cu").write_text("// quant\n", encoding="utf-8")
    (tmp_path / "gen_dequant_api.cu").write_text("// dequant\n", encoding="utf-8")

    missing = missing_generated_sources(tmp_path)

    assert missing == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_missing_generated_sources_reports_signature_only_cuda_files(tmp_path):
    (tmp_path / "gen_quant_api.cu").write_text("torch::Tensor quantize(", encoding="utf-8")
    (tmp_path / "gen_dequant_api.cu").write_text("torch::Tensor dequantize(", encoding="utf-8")

    missing = missing_generated_sources(tmp_path)

    assert missing == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_ensure_generated_sources_skips_when_generated_files_exist(tmp_path):
    (tmp_path / "gen_quant_api.cu").write_text(
        "torch::Tensor quantize(\nreturn output;",
        encoding="utf-8",
    )
    (tmp_path / "gen_dequant_api.cu").write_text(
        "torch::Tensor dequantize(\nreturn output;",
        encoding="utf-8",
    )

    result = ensure_generated_sources(tmp_path, run_generator=lambda command: None)

    assert result.generated is False
    assert result.sources == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_ensure_generated_sources_runs_generators_for_missing_files(tmp_path):
    commands = []

    def run_generator(command):
        commands.append(command)
        script = Path(command[1]).name
        if script == "gen_code_quant.py":
            (tmp_path / "gen_quant_api.cu").write_text(
                "torch::Tensor quantize(\nreturn output;",
                encoding="utf-8",
            )
        elif script == "gen_code_dequant.py":
            (tmp_path / "gen_dequant_api.cu").write_text(
                "torch::Tensor dequantize(\nreturn output;",
                encoding="utf-8",
            )

    result = ensure_generated_sources(tmp_path, run_generator=run_generator)

    assert result.generated is True
    assert len(commands) == 2
    assert result.sources == tuple(tmp_path / name for name in GENERATED_SOURCE_NAMES)


def test_ensure_generated_sources_raises_if_generator_does_not_create_files(tmp_path):
    with pytest.raises(RuntimeError, match="generated CUDA sources are still missing"):
        ensure_generated_sources(tmp_path, run_generator=lambda command: None)


@pytest.mark.parametrize("name", ("gen_code_quant.py", "gen_code_dequant.py"))
def test_cuda_generator_emits_device_guard_and_launch_check(name):
    source = (CUDA_SOURCE_DIR / name).read_text(encoding="utf-8")

    assert "#include <c10/cuda/CUDAGuard.h>" in source
    assert "#include <c10/cuda/CUDAException.h>" in source
    assert "c10::cuda::CUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK()" in source


@pytest.mark.parametrize(
    "name",
    ("quant_pack_kernel.cu", "dequant_reduce_kernel.cu"),
)
def test_handwritten_cuda_entries_guard_device_and_check_launch(name):
    source = (CUDA_SOURCE_DIR / name).read_text(encoding="utf-8")

    assert "#include <c10/cuda/CUDAGuard.h>" in source
    assert "#include <c10/cuda/CUDAException.h>" in source
    assert "c10::cuda::CUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK()" in source
