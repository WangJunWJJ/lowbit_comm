from __future__ import annotations

import ast
from pathlib import Path

import lowbit_comm


ROOT = Path(__file__).parents[3]


def test_public_api_snapshot_is_typed_and_versioned() -> None:
    assert set(lowbit_comm.__all__) == {
        "AutoAlgorithm",
        "BackendRegistry",
        "BenchmarkEvidence",
        "CommunicationProgram",
        "CompileContext",
        "CompressedAllGather",
        "CompressedReduceScatter",
        "CompressedReduceScatterAllGather",
        "DataType",
        "ErrorFeedbackDomain",
        "FullPrecisionWire",
        "FullTensor",
        "NativeAllReduce",
        "QuantizedWire",
        "ReduceMean",
        "ReduceSum",
        "ReducedShard",
        "RuntimeBindings",
        "__version__",
        "compile",
    }
    assert lowbit_comm.__version__ == "0.3.0"


def test_release_tree_contains_no_legacy_python_control_plane() -> None:
    assert not (ROOT / "ccdl_comm").exists()
    offenders: list[str] = []
    for directory in (
        ROOT / "src",
        ROOT / "examples",
        ROOT / "tests" / "contract",
    ):
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(name == "ccdl_comm" or name.startswith("ccdl_comm.") for name in names):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_root_build_only_packages_lowbit_comm() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "lowbit_comm"' in pyproject
    assert 'include = ["lowbit_comm*"]' in pyproject
    assert "ccdl_comm" not in pyproject
    assert "exclude-package-data" in pyproject
    assert "**/*.pyc" in pyproject
    assert '"csrc/**/*"' not in pyproject


def test_contract_suite_uses_responsibility_based_directory_name() -> None:
    assert (ROOT / "tests" / "contract").is_dir()
    assert not (ROOT / "tests" / "v03").exists()
