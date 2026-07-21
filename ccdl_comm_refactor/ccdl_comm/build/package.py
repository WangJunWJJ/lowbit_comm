from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .extension import create_cuda_extension


def package_csrc_root() -> Path:
    """Return the package-local CUDA source root."""

    return Path(__file__).resolve().parents[1] / "csrc"


def create_package_cuda_extension(
    *,
    name: str = "ccdl_cuda_ops",
    run_generator: Callable[[Sequence[str]], None] | None = None,
    extension_factory: Callable[..., object] | None = None,
) -> object:
    """Create the CCDL CUDA extension spec from package-local sources."""

    kwargs = {
        "name": name,
        "run_generator": run_generator,
    }
    if extension_factory is not None:
        kwargs["extension_factory"] = extension_factory
    return create_cuda_extension(package_csrc_root(), **kwargs)
