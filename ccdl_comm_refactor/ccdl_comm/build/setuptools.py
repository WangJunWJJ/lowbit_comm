from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from .package import create_package_cuda_extension


def _cuda_build_requested(env: Mapping[str, str]) -> bool:
    return env.get("CCDL_COMM_BUILD_CUDA", "").strip().lower() in {"1", "true", "yes", "on"}


def _torch_build_ext_class():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension


def build_setup_kwargs(
    *,
    env: Mapping[str, str],
    create_extension: Callable[[], object] = create_package_cuda_extension,
    build_ext_class: Callable[[], type] = _torch_build_ext_class,
) -> dict[str, object]:
    """Return setuptools kwargs for optional CUDA extension builds.

    CUDA compilation is opt-in so metadata commands and CPU-only imports remain
    safe when PyTorch or a compiler toolchain is unavailable.
    """

    if not _cuda_build_requested(env):
        return {"ext_modules": [], "cmdclass": {}}

    return {
        "ext_modules": [create_extension()],
        "cmdclass": {"build_ext": build_ext_class()},
    }
