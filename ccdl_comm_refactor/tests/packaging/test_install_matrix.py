from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.packaging.wheel_helpers import build_wheel


def _install(wheels: tuple[Path, ...], target: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            *(str(wheel) for wheel in wheels),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_isolated(target: Path, code: str, cwd: Path) -> str:
    command = f"import sys; sys.path.insert(0, {str(target)!r}); {code}"
    result = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_core_only_install_imports_without_torch(tmp_path) -> None:
    core = build_wheel("ccdl-core", tmp_path / "core-wheel")
    target = tmp_path / "site"
    _install((core,), target)

    output = _run_isolated(
        target,
        "import ccdl_comm; print(ccdl_comm.CCDL_CORE_ABI)",
        tmp_path,
    )

    assert output == "1"


def test_backend_metadata_wheels_preserve_missing_extension_diagnostics(tmp_path) -> None:
    core = build_wheel("ccdl-core", tmp_path / "core-wheel")
    cuda = build_wheel("ccdl-cuda", tmp_path / "cuda-wheel")
    ascend = build_wheel("ccdl-ascend", tmp_path / "ascend-wheel")
    target = tmp_path / "site"
    _install((core, cuda, ascend), target)

    output = _run_isolated(
        target,
        "from ccdl_comm.cuda.loader import load_cuda_extension; "
        "from ccdl_comm.ascend.loader import load_cann_extension; "
        "cuda = load_cuda_extension(); cann = load_cann_extension(); "
        "print(cuda.available, cann.available, bool(cuda.reason), bool(cann.reason))",
        tmp_path,
    )

    assert output == "False False True True"
