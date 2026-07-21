from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence


GENERATED_SOURCE_NAMES = ("gen_quant_api.cu", "gen_dequant_api.cu")
_GENERATED_SOURCE_MARKERS = {
    "gen_quant_api.cu": ("torch::Tensor quantize(", "return output;"),
    "gen_dequant_api.cu": ("torch::Tensor dequantize(", "return output;"),
}


@dataclass(frozen=True)
class CodegenResult:
    """Result of validating or generating CUDA source files."""

    generated: bool
    sources: tuple[Path, ...]


def missing_generated_sources(source_dir: str | Path) -> tuple[Path, ...]:
    """Return generated CUDA files that are absent or do not define the API."""

    root = Path(source_dir)
    missing = []
    for name in GENERATED_SOURCE_NAMES:
        path = root / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker not in text for marker in _GENERATED_SOURCE_MARKERS[name]):
            missing.append(path)
    return tuple(missing)


def _run_generator(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def ensure_generated_sources(
    source_dir: str | Path,
    *,
    run_generator: Callable[[Sequence[str]], None] = _run_generator,
) -> CodegenResult:
    """Ensure CCDL generated CUDA translation units exist.

    The original CCDL prototype requires running two generator scripts before
    building the extension.  This helper makes that requirement explicit and
    testable so packaging can call it before extension compilation.
    """

    root = Path(source_dir)
    expected = tuple(root / name for name in GENERATED_SOURCE_NAMES)
    if not missing_generated_sources(root):
        return CodegenResult(generated=False, sources=expected)

    generator_scripts = ("gen_code_quant.py", "gen_code_dequant.py")
    for script_name in generator_scripts:
        script = root / script_name
        command = (sys.executable, str(script), "--output-dir-path", str(root))
        run_generator(command)

    missing = missing_generated_sources(root)
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise RuntimeError(f"generated CUDA sources are still missing: {missing_names}")

    return CodegenResult(generated=True, sources=expected)
