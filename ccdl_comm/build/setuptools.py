from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from .cann import create_cann_extension as create_package_cann_extension
from .package import create_package_cuda_extension


def _truthy_env(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _truthy_env_with_legacy_alias(
    env: Mapping[str, str],
    canonical_name: str,
    legacy_name: str,
) -> bool:
    """Resolve a canonical build flag before its legacy compatibility alias."""

    if canonical_name in env:
        return _truthy_env(env, canonical_name)
    return _truthy_env(env, legacy_name)


def _torch_build_ext_class():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension


def build_setup_kwargs(
    *,
    env: Mapping[str, str],
    create_extension: Callable[[], object] = create_package_cuda_extension,
    create_cann_extension: Callable[[], object] = create_package_cann_extension,
    build_ext_class: Callable[[], type] = _torch_build_ext_class,
) -> dict[str, object]:
    """Return setuptools kwargs for optional native extension builds.

    Native compilation is opt-in so metadata commands and CPU-only imports
    remain safe when PyTorch, torch-npu, or compiler toolchains are unavailable.
    """

    kwargs: dict[str, object] = {
        "name": "ccdl-comm",
        "version": "0.1.0",
        "description": "Low-bit compressed communication library for ParaScale native-DDP integration.",
        "packages": [
            "ccdl_comm",
            "ccdl_comm.ascend",
            "ccdl_comm.build",
            "ccdl_comm.collectives",
            "ccdl_comm.communication",
            "ccdl_comm.cuda",
            "ccdl_comm.cuda.transports",
            "ccdl_comm.quantization",
        ],
        "include_package_data": True,
    }

    ext_modules = []
    if _truthy_env_with_legacy_alias(
        env,
        "CCDL_COMM_BUILD_CUDA",
        "CCDL_BUILD_CUDA",
    ):
        ext_modules.append(create_extension())
    if _truthy_env(env, "CCDL_COMM_BUILD_CANN"):
        ext_modules.append(create_cann_extension())

    if not ext_modules:
        kwargs.update({"ext_modules": [], "cmdclass": {}})
        return kwargs

    kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext_class()},
        }
    )
    return kwargs
