from __future__ import annotations

from tests.packaging.wheel_helpers import build_wheel, wheel_files, wheel_metadata


def _requirements(wheel) -> tuple[str, ...]:
    return tuple(wheel_metadata(wheel).get_all("Requires-Dist", []))


def test_cuda_wheel_declares_core_abi_compatible_runtime(tmp_path) -> None:
    wheel = build_wheel("ccdl-cuda", tmp_path / "wheel")
    requirements = _requirements(wheel)

    assert "ccdl-core==0.1.0" in requirements
    assert any(requirement.lower().startswith("torch") for requirement in requirements)


def test_cuda_wheel_does_not_include_core_or_cann_sources(tmp_path) -> None:
    files = wheel_files(build_wheel("ccdl-cuda", tmp_path / "wheel"))

    assert "ccdl_comm/__init__.py" not in files
    assert not any("csrc_ascend" in name or "/ascend/" in name for name in files)


def test_ascend_wheel_declares_core_abi_compatible_runtime(tmp_path) -> None:
    wheel = build_wheel("ccdl-ascend", tmp_path / "wheel")
    requirements = _requirements(wheel)

    assert "ccdl-core==0.1.0" in requirements
    assert any(requirement.lower().startswith("torch") for requirement in requirements)
    assert any(requirement.lower().startswith("torch-npu") for requirement in requirements)


def test_ascend_wheel_does_not_include_core_or_cuda_sources(tmp_path) -> None:
    files = wheel_files(build_wheel("ccdl-ascend", tmp_path / "wheel"))

    assert "ccdl_comm/__init__.py" not in files
    assert not any("/csrc/" in name or "/cuda/" in name for name in files)
