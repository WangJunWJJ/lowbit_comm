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
