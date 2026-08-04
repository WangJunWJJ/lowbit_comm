from __future__ import annotations

from pathlib import Path


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").is_dir():
            return candidate
    raise RuntimeError("cannot locate lowbit_comm Git root")


def test_refactored_project_is_the_repository_root() -> None:
    root = _repository_root(Path(__file__).resolve())

    assert (root / "ccdl_comm" / "__init__.py").is_file()
    assert (root / "packages" / "ccdl-core" / "pyproject.toml").is_file()
    assert (root / "tests" / "packaging" / "test_core_wheel.py").is_file()
    assert (root / "CONTRIBUTING.md").is_file()
    assert (root / "README.md").read_text(encoding="utf-8").splitlines()[0] == "# lowbit_comm"


def test_legacy_and_staging_trees_are_absent() -> None:
    root = _repository_root(Path(__file__).resolve())

    for relative in (
        "ccdl_comm_refactor",
        "ccdl",
        "csrc",
        "benchmarks",
        "CODE_AUDIT_REPORT.md",
    ):
        assert not (root / relative).exists(), relative
