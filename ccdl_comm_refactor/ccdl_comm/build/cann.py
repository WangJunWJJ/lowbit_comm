from __future__ import annotations

import os
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


def _torch_npu_include_root() -> Path:
    import torch_npu

    return Path(torch_npu.__file__).resolve().parent / "include"


def _torch_npu_library_root() -> Path:
    import torch_npu

    return Path(torch_npu.__file__).resolve().parent / "lib"


def _cann_include_root() -> Path:
    return Path("/usr/local/Ascend/cann-9.0.0/aarch64-linux/include")


def create_cann_extension(
    csrc_root: str | Path | None = None,
    *,
    name: str = "ccdl_cann_ops",
    extension_factory: Callable[..., object] = _default_extension_factory,
    torch_npu_include_root: Callable[[], Path] = _torch_npu_include_root,
    torch_npu_library_root: Callable[[], Path] = _torch_npu_library_root,
    cann_include_root: Callable[[], Path] = _cann_include_root,
) -> object:
    """Create a CANN extension spec from package-local sources."""

    root = Path(csrc_root) if csrc_root is not None else package_csrc_ascend_root()
    cann_include = cann_include_root()
    experimental_aclnn = os.environ.get("CCDL_COMM_EXPERIMENTAL_ACLNN") == "1"

    include_dirs = [str(cann_include)]
    libraries: list[str] = []
    library_dirs: list[str] = []
    runtime_library_dirs: list[str] = []
    extra_compile_args = ["-O3"]

    if experimental_aclnn:
        torch_npu_include = torch_npu_include_root()
        torch_npu_library = torch_npu_library_root()
        include_dirs.append(str(torch_npu_include / "third_party" / "op-plugin"))
        libraries.append("torch_npu")
        library_dirs.append(str(torch_npu_library))
        runtime_library_dirs.append(str(torch_npu_library))
        extra_compile_args.append("-DCCDL_COMM_EXPERIMENTAL_ACLNN")

    return extension_factory(
        name=name,
        sources=[str(path) for path in collect_cann_sources(root)],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        extra_compile_args=extra_compile_args,
    )
