from __future__ import annotations

from pathlib import Path

import ccdl_comm.build.distributions as distributions
from packaging.requirements import Requirement

from tests.packaging.wheel_helpers import build_wheel, wheel_files, wheel_metadata


def _requirements(wheel) -> tuple[str, ...]:
    return tuple(wheel_metadata(wheel).get_all("Requires-Dist", []))


def _has_requirement(requirements: tuple[str, ...], name: str, specifier: str) -> bool:
    return any(
        parsed.name == name and str(parsed.specifier) == specifier
        for parsed in map(Requirement, requirements)
    )


def test_cuda_wheel_declares_core_abi_compatible_runtime(tmp_path) -> None:
    wheel = build_wheel("ccdl-cuda", tmp_path / "wheel")
    requirements = _requirements(wheel)

    assert _has_requirement(requirements, "ccdl-core", "==0.1.0")
    assert any(requirement.lower().startswith("torch") for requirement in requirements)


def test_cuda_wheel_does_not_include_core_or_cann_sources(tmp_path) -> None:
    files = wheel_files(build_wheel("ccdl-cuda", tmp_path / "wheel"))

    assert "ccdl_comm/__init__.py" not in files
    assert not any("csrc_ascend" in name or "/ascend/" in name for name in files)


def test_ascend_wheel_declares_core_abi_compatible_runtime(tmp_path) -> None:
    wheel = build_wheel("ccdl-ascend", tmp_path / "wheel")
    requirements = _requirements(wheel)

    assert _has_requirement(requirements, "ccdl-core", "==0.1.0")
    assert any(requirement.lower().startswith("torch") for requirement in requirements)
    assert any(requirement.lower().startswith("torch-npu") for requirement in requirements)


def test_ascend_wheel_does_not_include_core_or_cuda_sources(tmp_path) -> None:
    files = wheel_files(build_wheel("ccdl-ascend", tmp_path / "wheel"))

    assert "ccdl_comm/__init__.py" not in files
    assert not any("/csrc/" in name or "/cuda/" in name for name in files)


def test_backend_build_ext_creates_empty_build_temp_before_compiling(tmp_path) -> None:
    calls = []

    class FakeBuildExt:
        build_temp = str(tmp_path / "missing" / "temp")

        def build_extensions(self):
            calls.append("compiled")

    command = distributions._safe_build_ext_class(FakeBuildExt)()
    command.build_extensions()

    assert Path(command.build_temp).is_dir()
    assert calls == ["compiled"]


def test_cuda_extension_uses_absolute_shared_sources(tmp_path) -> None:
    package_root = tmp_path / "repository" / "packages" / "ccdl-cuda"
    package_root.mkdir(parents=True)
    csrc = tmp_path / "repository" / "ccdl_comm" / "csrc"
    (csrc / "executor").mkdir(parents=True)
    (csrc / "quantization").mkdir()
    (csrc / "pybind.cpp").write_text("// binding", encoding="utf-8")
    for name in ("quantization.cpp", "quantization.cu"):
        (csrc / "quantization" / name).write_text("// source", encoding="utf-8")
    (csrc / "quantization" / "gen_quant_api.cu").write_text(
        "torch::Tensor quantize() { return output; }",
        encoding="utf-8",
    )
    (csrc / "quantization" / "gen_dequant_api.cu").write_text(
        "torch::Tensor dequantize() { return output; }",
        encoding="utf-8",
    )

    kwargs = distributions.cuda_setup_kwargs(
        package_root,
        {"CCDL_COMM_BUILD_CUDA": "1"},
        extension_factory=lambda **values: values,
        build_ext_class=lambda: object,
    )

    assert all(Path(source).is_absolute() for source in kwargs["ext_modules"][0]["sources"])
