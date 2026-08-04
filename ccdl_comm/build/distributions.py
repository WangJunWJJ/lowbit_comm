"""Monorepo build definitions for independently published CCDL wheels."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from .cann import create_cann_extension
from .extension import create_cuda_extension


CCDL_VERSION = "0.1.0"
CORE_DISTRIBUTION = f"ccdl-core=={CCDL_VERSION}"
CORE_PACKAGES = (
    "ccdl_comm",
    "ccdl_comm.ascend",
    "ccdl_comm.build",
    "ccdl_comm.collectives",
    "ccdl_comm.communication",
    "ccdl_comm.cuda",
    "ccdl_comm.cuda.transports",
    "ccdl_comm.quantization",
)


def core_setup_kwargs(package_root: Path) -> dict[str, object]:
    """Map the core distribution to the repository's unique Python sources."""

    return {
        "packages": list(CORE_PACKAGES),
        "package_dir": {
            "": Path(
                os.path.relpath(_repository_from(package_root), package_root)
            ).as_posix()
        },
        "include_package_data": False,
    }


def cuda_setup_kwargs(
    package_root: Path,
    env: Mapping[str, str],
    *,
    extension_factory: Callable[..., object] | None = None,
    build_ext_class: Callable[[], type] | None = None,
) -> dict[str, object]:
    """Build only the CUDA binary module when explicitly requested."""

    return _backend_setup_kwargs(
        package_root,
        enabled=_flag(env, "CCDL_COMM_BUILD_CUDA", "CCDL_BUILD_CUDA"),
        extension_builder=lambda repository: _cuda_extension(
            repository,
            extension_factory,
        ),
        build_ext_class=build_ext_class,
    )


def ascend_setup_kwargs(
    package_root: Path,
    env: Mapping[str, str],
    *,
    extension_factory: Callable[..., object] | None = None,
    build_ext_class: Callable[[], type] | None = None,
) -> dict[str, object]:
    """Build only the Ascend binary module when explicitly requested."""

    return _backend_setup_kwargs(
        package_root,
        enabled=_flag(env, "CCDL_COMM_BUILD_CANN", "CCDL_BUILD_CANN"),
        extension_builder=lambda repository: _ascend_extension(
            repository,
            extension_factory,
        ),
        build_ext_class=build_ext_class,
    )


def _backend_setup_kwargs(
    package_root: Path,
    *,
    enabled: bool,
    extension_builder: Callable[[Path], object],
    build_ext_class: Callable[[], type] | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "packages": [],
        "py_modules": [],
        "ext_modules": [],
        "cmdclass": {},
    }
    if not enabled:
        return kwargs
    extension = extension_builder(_repository_from(package_root))
    if build_ext_class is None:
        from torch.utils.cpp_extension import BuildExtension

        base_build_ext = BuildExtension
    else:
        base_build_ext = build_ext_class()

    kwargs.update(
        ext_modules=[extension],
        cmdclass={"build_ext": _safe_build_ext_class(base_build_ext)},
    )
    return kwargs


def _cuda_extension(
    repository: Path,
    extension_factory: Callable[..., object] | None,
) -> object:
    kwargs: dict[str, object] = {}
    if extension_factory is not None:
        kwargs["extension_factory"] = extension_factory
    extension = create_cuda_extension(
        repository / "ccdl_comm" / "csrc",
        source_base=repository,
        **kwargs,
    )
    _make_sources_absolute(extension, repository)
    return extension


def _ascend_extension(
    repository: Path,
    extension_factory: Callable[..., object] | None,
) -> object:
    kwargs: dict[str, object] = {}
    if extension_factory is not None:
        kwargs["extension_factory"] = extension_factory
    extension = create_cann_extension(
        repository / "ccdl_comm" / "csrc_ascend",
        **kwargs,
    )
    _make_sources_absolute(extension, repository)
    return extension


def _make_sources_absolute(extension: object, repository: Path) -> None:
    sources = getattr(extension, "sources", None)
    if sources is None and isinstance(extension, dict):
        sources = extension.get("sources")
    if sources is None:
        raise TypeError("extension must expose mutable sources")
    absolute_sources = []
    for source in sources:
        path = Path(source)
        absolute = path if path.is_absolute() else repository / path
        absolute_sources.append(str(absolute.resolve()))
    if isinstance(extension, dict):
        extension["sources"] = absolute_sources
    else:
        extension.sources = absolute_sources


def _safe_build_ext_class(base: type) -> type:
    """Ensure Torch's Ninja writer always receives an existing build temp."""

    class SafeBackendBuildExt(base):
        def build_extensions(self):
            Path(self.build_temp).mkdir(parents=True, exist_ok=True)
            return super().build_extensions()

    SafeBackendBuildExt.__name__ = f"Safe{base.__name__}"
    return SafeBackendBuildExt


def _repository_from(package_root: Path) -> Path:
    return package_root.resolve().parents[1]


def _flag(env: Mapping[str, str], canonical: str, legacy: str) -> bool:
    name = canonical if canonical in env else legacy
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "CCDL_VERSION",
    "CORE_DISTRIBUTION",
    "CORE_PACKAGES",
    "ascend_setup_kwargs",
    "core_setup_kwargs",
    "cuda_setup_kwargs",
]
