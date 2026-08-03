from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_wheel(package: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("CCDL_COMM_BUILD_CUDA", None)
    environment.pop("CCDL_BUILD_CUDA", None)
    environment.pop("CCDL_COMM_BUILD_CANN", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(output),
            str(PROJECT_ROOT / "packages" / package),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def wheel_files(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        return tuple(sorted(archive.namelist()))


def wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        return BytesParser().parsebytes(archive.read(metadata_name))
