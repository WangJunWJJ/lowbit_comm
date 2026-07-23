from __future__ import annotations

from pathlib import Path
from typing import Callable


def collect_cann_sources(csrc_root: str | Path) -> tuple[Path, ...]:
    """Collect CANN extension translation units in deterministic order."""

    root = Path(csrc_root)
    sources = [root / "pybind.cpp"]
    sources.extend(sorted(path for path in root.glob("*.cpp") if path.name != "pybind.cpp"))
    sources.extend(sorted((root / "kernels").glob("*.cpp")) if (root / "kernels").exists() else [])
    return tuple(sources)


def _default_extension_factory(**kwargs):
    from torch_npu.utils.cpp_extension import NpuExtension

    return NpuExtension(**kwargs)


def package_csrc_ascend_root() -> Path:
    """Return the package-local Ascend CANN source root."""

    return Path(__file__).resolve().parents[1] / "csrc_ascend"


def create_cann_extension(
    csrc_root: str | Path | None = None,
    *,
    name: str = "ccdl_cann_ops",
    extension_factory: Callable[..., object] = _default_extension_factory,
) -> object:
    """Create a CANN extension spec from package-local sources."""

    root = Path(csrc_root) if csrc_root is not None else package_csrc_ascend_root()
    return extension_factory(
        name=name,
        sources=[str(path) for path in collect_cann_sources(root)],
        extra_compile_args=["-O3"],
    )
