"""Build specification for the package-local CUDA extension."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


CSRC_ROOT = Path(__file__).with_name("csrc")
GENERATED_SOURCES = ("gen_quant_api.cu", "gen_dequant_api.cu")


def ensure_generated_sources(source_dir: Path) -> None:
    """Generate missing API translation units before source collection."""

    missing = [name for name in GENERATED_SOURCES if not (source_dir / name).is_file()]
    if not missing:
        return
    for script in ("gen_code_quant.py", "gen_code_dequant.py"):
        subprocess.run(
            [
                sys.executable,
                str(source_dir / script),
                "--output-dir-path",
                str(source_dir),
            ],
            check=True,
        )
    still_missing = [name for name in GENERATED_SOURCES if not (source_dir / name).is_file()]
    if still_missing:
        raise RuntimeError(
            "CUDA code generation did not produce: " + ", ".join(still_missing)
        )


def create_cuda_extension(
    *,
    extension_factory: Callable[..., Any] | None = None,
    ensure_generated: Callable[[Path], None] = ensure_generated_sources,
) -> Any:
    """Create a deterministic CUDAExtension spec without importing Torch eagerly."""

    quantization = CSRC_ROOT / "quantization"
    ensure_generated(quantization)
    sources = [CSRC_ROOT / "pybind.cpp"]
    sources.extend((CSRC_ROOT / "executor").glob("*.cpp"))
    sources.extend(quantization.glob("*.cu"))
    source_names = sorted(str(path) for path in sources)

    if extension_factory is None:
        from torch.utils.cpp_extension import CUDAExtension

        extension_factory = CUDAExtension
    return extension_factory(
        name="lowbit_comm_cuda_ops",
        sources=source_names,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-U__CUDA_NO_HALF_OPERATORS__"],
        },
    )
