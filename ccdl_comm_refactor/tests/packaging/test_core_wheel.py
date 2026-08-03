from __future__ import annotations

import tomllib

import ccdl_comm

from tests.packaging.wheel_helpers import PROJECT_ROOT, build_wheel, wheel_files, wheel_metadata


def test_core_abi_is_public_and_stable() -> None:
    assert ccdl_comm.CCDL_CORE_ABI == 1


def test_core_metadata_has_no_torch_dependency(tmp_path) -> None:
    wheel = build_wheel("ccdl-core", tmp_path / "wheel")
    requires = wheel_metadata(wheel).get_all("Requires-Dist", [])

    assert not any(requirement.lower().startswith("torch") for requirement in requires)


def test_core_wheel_owns_python_sources_without_native_sources(tmp_path) -> None:
    files = wheel_files(build_wheel("ccdl-core", tmp_path / "wheel"))

    assert "ccdl_comm/__init__.py" in files
    assert "ccdl_comm/abi.py" in files
    assert "ccdl_comm/cuda/loader.py" in files
    assert "ccdl_comm/ascend/loader.py" in files
    assert not any("/csrc/" in name or "/csrc_ascend/" in name for name in files)
    assert not any(name.endswith((".so", ".pyd")) for name in files)


def test_root_distribution_is_a_core_compatibility_meta_package() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["dependencies"] == ["ccdl-core==0.1.0"]
    assert metadata["tool"]["setuptools"]["packages"] == []
