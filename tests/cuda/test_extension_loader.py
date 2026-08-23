"""Contract tests for the optional CUDA extension boundary."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lowbit_comm.core.errors import CompileError


ROOT = Path(__file__).resolve().parents[2]


def _isolated_env() -> dict[str, str]:
    """Keep the isolated child process free from user Python configuration."""
    return {"PATH": os.environ["PATH"]}


def test_top_level_import_does_not_load_torch_or_cuda_extension() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import lowbit_comm; "
        "assert 'torch' not in sys.modules; "
        "assert 'lowbit_comm._C' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        env=_isolated_env(),
    )


def test_loader_rejects_wrong_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = importlib.reload(
        importlib.import_module("lowbit_comm.backends.cuda.loader")
    )
    fake = SimpleNamespace(abi_version=lambda: 2)
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)

    with pytest.raises(CompileError, match="CUDA extension ABI"):
        loader.load_extension()

    assert loader.extension_available() is False
    assert loader.extension_error() == "CUDA extension ABI is incompatible."


def test_loader_reports_missing_extension_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = importlib.reload(
        importlib.import_module("lowbit_comm.backends.cuda.loader")
    )
    missing = ModuleNotFoundError("lowbit_comm._C")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(missing),
    )

    with pytest.raises(
        CompileError,
        match="CUDA extension is unavailable.",
    ) as caught:
        loader.load_extension()

    with pytest.raises(CompileError) as cached:
        loader.load_extension()

    assert cached.value is caught.value
    assert loader.extension_available() is False
    assert loader.extension_error() == "CUDA extension is unavailable."


def test_cuda_build_entrypoint_lists_only_build_generated_sources() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")
    assert 'name="lowbit_comm._C"' in setup_source
    assert 'BUILD_DIR / "quantization" / "gen_quant_api.cu"' in setup_source
    assert 'BUILD_DIR / "quantization" / "gen_dequant_api.cu"' in setup_source
    assert 'CSRC_DIR / "quantization" / "gen_quant_api.cu"' not in setup_source
    assert (
        'CSRC_DIR / "quantization" / "gen_dequant_api.cu"' not in setup_source
    )


def test_cuda_build_entrypoint_includes_quantization_headers() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert 'str(CSRC_DIR / "quantization")' in setup_source


def test_cuda_build_entrypoint_includes_private_qwd_sources() -> None:
    setup_source = (ROOT / "setup_cuda.py").read_text(encoding="utf-8")

    assert 'CSRC_DIR / "executor" / "qwd_plan.cpp"' in setup_source
    assert (
        'CSRC_DIR / "quantization" / "qwd_restore_kernel.cu"' in setup_source
    )
    assert '"-DUSE_C10D_NCCL"' in setup_source


def test_dequant_sum_does_not_require_half_operator_overloads() -> None:
    kernel_source = (
        ROOT / "csrc" / "quantization" / "dequant_kernel.cuh"
    ).read_text(encoding="utf-8")

    assert "glb[index] = __hadd(glb[index], srd[index]);" in kernel_source
    assert "glb[index] += srd[index];" not in kernel_source


def test_cuda_abi_is_bound_from_the_shared_runtime_header() -> None:
    abi_header = (ROOT / "csrc" / "runtime" / "abi.h").read_text(
        encoding="utf-8"
    )
    pybind_source = (ROOT / "csrc" / "pybind.cpp").read_text(encoding="utf-8")

    assert "kLowbitCommCudaAbiVersion = 1" in abi_header
    assert '"abi_version"' in pybind_source
    assert "kLowbitCommCudaAbiVersion" in pybind_source
