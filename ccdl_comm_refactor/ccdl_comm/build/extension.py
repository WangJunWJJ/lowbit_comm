from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .codegen import ensure_generated_sources


def collect_cuda_sources(csrc_root: str | Path) -> tuple[Path, ...]:
    """Collect CUDA extension translation units in deterministic order.

    Args:
        csrc_root: Root directory containing `pybind.cpp` and `quantization/`.

    Returns:
        Tuple of source paths sorted for reproducible extension builds.
    """

    root = Path(csrc_root)
    sources = [root / "pybind.cpp"]
    sources.extend(sorted((root / "quantization").glob("*.cu")))
    sources.extend(sorted((root / "quantization").glob("*.cpp")))
    return tuple(sources)


def _default_extension_factory(**kwargs):
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(**kwargs)


def create_cuda_extension(
    csrc_root: str | Path,
    *,
    name: str = "ccdl_cuda_ops",
    run_generator: Callable[[Sequence[str]], None] | None = None,
    extension_factory: Callable[..., object] = _default_extension_factory,
) -> object:
    """Create a CUDA extension spec after ensuring generated sources exist.

    Args:
        csrc_root: Root directory containing CCDL C++/CUDA sources.
        name: Python extension module name.
        run_generator: Optional code-generation runner used by tests and build
            backends. When omitted, the default generator runner is used.
        extension_factory: Factory compatible with `CUDAExtension`.

    Returns:
        Extension object created by `extension_factory`.
    """

    root = Path(csrc_root)
    quantization_dir = root / "quantization"
    if run_generator is None:
        ensure_generated_sources(quantization_dir)
    else:
        ensure_generated_sources(quantization_dir, run_generator=run_generator)

    sources = [str(path) for path in collect_cuda_sources(root)]
    return extension_factory(
        name=name,
        sources=sources,
        extra_compile_args={
            "cxx": [],
            "nvcc": ["-O3", "-U__CUDA_NO_HALF_OPERATORS__"],
        },
    )
