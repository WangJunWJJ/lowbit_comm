"""Build the optional lowbit_comm CUDA extension with deterministic inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
BUILD_DIR = ROOT / "build"
GENERATED_DIR = BUILD_DIR / "quantization"

GENERATED_SOURCES = (
    BUILD_DIR / "quantization" / "gen_quant_api.cu",
    BUILD_DIR / "quantization" / "gen_dequant_api.cu",
)
SOURCES = (
    CSRC_DIR / "pybind.cpp",
    CSRC_DIR / "executor" / "compressed_work.cpp",
    CSRC_DIR / "executor" / "cuda_executor.cpp",
    CSRC_DIR / "executor" / "fulltensor_plan.cpp",
    CSRC_DIR / "executor" / "reduced_shard_plan.cpp",
    CSRC_DIR / "runtime" / "workspace_pool.cpp",
    GENERATED_SOURCES[0],
    GENERATED_SOURCES[1],
    CSRC_DIR / "quantization" / "quant_pack_kernel.cu",
    CSRC_DIR / "quantization" / "dequant_reduce_kernel.cu",
    CSRC_DIR / "quantization" / "utils.cu",
)


def generate_sources() -> None:
    """Generate CUDA instantiations outside the tracked source tree."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for generator in (
        CSRC_DIR / "quantization" / "gen_code_quant.py",
        CSRC_DIR / "quantization" / "gen_code_dequant.py",
    ):
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--output-dir-path",
                str(GENERATED_DIR),
            ],
            check=True,
        )


generate_sources()

setup(
    name="lowbit_comm_cuda",
    ext_modules=[
        CUDAExtension(
            name="lowbit_comm._C",
            sources=[str(source) for source in SOURCES],
            include_dirs=[
                str(CSRC_DIR),
                str(CSRC_DIR / "quantization"),
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
