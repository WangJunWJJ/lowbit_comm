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
) -> dict[str, object]:
    """Build only the CUDA binary module when explicitly requested."""

    return _backend_setup_kwargs(
        package_root,
        enabled=_flag(env, "CCDL_COMM_BUILD_CUDA", "CCDL_BUILD_CUDA"),
        extension_builder=lambda repository: _cuda_extension(
            repository,
            package_root,
            extension_factory,
        ),
    )


def ascend_setup_kwargs(
    package_root: Path,
    env: Mapping[str, str],
    *,
    extension_factory: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Build only the Ascend binary module when explicitly requested."""

    return _backend_setup_kwargs(
        package_root,
        enabled=_flag(env, "CCDL_COMM_BUILD_CANN", "CCDL_BUILD_CANN"),
        extension_builder=lambda repository: _ascend_extension(
            repository,
            package_root,
            extension_factory,
        ),
    )


def _backend_setup_kwargs(
    package_root: Path,
    *,
    enabled: bool,
    extension_builder: Callable[[Path], object],
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
    from torch.utils.cpp_extension import BuildExtension

    kwargs.update(
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension},
    )
    return kwargs


def _cuda_extension(
    repository: Path,
    package_root: Path,
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
    _rebase_sources(extension, package_root, repository)
    return extension


def _ascend_extension(
    repository: Path,
    package_root: Path,
    extension_factory: Callable[..., object] | None,
) -> object:
    kwargs: dict[str, object] = {}
    if extension_factory is not None:
        kwargs["extension_factory"] = extension_factory
    extension = create_cann_extension(
        repository / "ccdl_comm" / "csrc_ascend",
        **kwargs,
    )
    _rebase_sources(extension, package_root, repository)
    return extension


def _rebase_sources(extension: object, package_root: Path, repository: Path) -> None:
    sources = getattr(extension, "sources", None)
    if sources is None and isinstance(extension, dict):
        sources = extension.get("sources")
    if sources is None:
        raise TypeError("extension must expose mutable sources")
    rebased = []
    for source in sources:
        path = Path(source)
        absolute = path if path.is_absolute() else repository / path
        rebased.append(Path(os.path.relpath(absolute, package_root)).as_posix())
    if isinstance(extension, dict):
        extension["sources"] = rebased
    else:
        extension.sources = rebased


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
