import subprocess
import sys
from pathlib import Path

import pytest

from ccdl_comm.build.setuptools import build_setup_kwargs


def test_setup_kwargs_disable_cuda_extension_by_default() -> None:
    kwargs = build_setup_kwargs(env={})

    assert kwargs["name"] == "ccdl-comm"
    assert "ccdl_comm.ascend" in kwargs["packages"]
    assert "ccdl_comm.collectives" in kwargs["packages"]
    assert "ccdl_comm.cuda.transports" in kwargs["packages"]
    assert kwargs["ext_modules"] == []
    assert kwargs["cmdclass"] == {}


def test_setup_kwargs_enable_cuda_extension_when_requested() -> None:
    calls = []

    def fake_create_extension():
        calls.append("called")
        return "extension"

    kwargs = build_setup_kwargs(
        env={"CCDL_COMM_BUILD_CUDA": "1"},
        create_extension=fake_create_extension,
        build_ext_class=lambda: "build_ext",
    )

    assert kwargs["ext_modules"] == ["extension"]
    assert calls == ["called"]
    assert "build_ext" in kwargs["cmdclass"]


def test_setup_kwargs_accept_legacy_cuda_build_flag() -> None:
    calls = []

    kwargs = build_setup_kwargs(
        env={"CCDL_BUILD_CUDA": "1"},
        create_extension=lambda: calls.append("called") or "extension",
        build_ext_class=lambda: "build_ext",
    )

    assert kwargs["ext_modules"] == ["extension"]
    assert calls == ["called"]


def test_canonical_cuda_build_flag_overrides_legacy_alias() -> None:
    calls = []

    kwargs = build_setup_kwargs(
        env={
            "CCDL_COMM_BUILD_CUDA": "0",
            "CCDL_BUILD_CUDA": "1",
        },
        create_extension=lambda: calls.append("called") or "extension",
        build_ext_class=lambda: "build_ext",
    )

    assert kwargs["ext_modules"] == []
    assert calls == []


def test_setup_kwargs_enable_cann_extension_when_requested() -> None:
    calls = []

    def fake_create_cann_extension():
        calls.append("called")
        return "cann-extension"

    kwargs = build_setup_kwargs(
        env={"CCDL_COMM_BUILD_CANN": "1"},
        create_cann_extension=fake_create_cann_extension,
        build_ext_class=lambda: "build_ext",
    )

    assert kwargs["ext_modules"] == ["cann-extension"]
    assert calls == ["called"]
    assert "build_ext" in kwargs["cmdclass"]


def test_setup_py_metadata_does_not_require_cuda_build() -> None:
    pytest.importorskip("setuptools")
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ccdl-comm"


def test_setup_py_loads_when_source_root_is_absent_from_sys_path(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    setup_path = project_root / "setup.py"
    command = f"""
import runpy
import sys
import types
from pathlib import Path

fake_setuptools = types.ModuleType("setuptools")
fake_setuptools.setup = lambda **kwargs: None
sys.modules["setuptools"] = fake_setuptools
project_root = Path({str(project_root)!r}).resolve()
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != project_root
]
runpy.run_path({str(setup_path)!r}, run_name="not_main")
"""

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
