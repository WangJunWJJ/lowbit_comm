from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "src" / "lowbit_comm"
CONTRACT = ROOT / "docs" / "architecture" / "architecture_contract.json"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(f"{forbidden}.")


def architecture_violations() -> list[str]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    violations: list[str] = []
    if not SOURCE.exists():
        return ["missing src/lowbit_comm"]
    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE)
        layer = relative.parts[0] if len(relative.parts) > 1 else "public"
        forbidden: tuple[str, ...] = ("ccdl_comm",)
        if layer == "core":
            forbidden += tuple(contract["core_forbidden_imports"])
        elif layer == "backends":
            forbidden += tuple(contract["backend_forbidden_imports"])
            forbidden += ("lowbit_comm.compiler",)
        for imported in _imports(path):
            if any(_matches(imported, item) for item in forbidden):
                violations.append(f"{relative.as_posix()}: {imported}")
    return sorted(set(violations))


def test_new_source_never_imports_legacy_control_plane() -> None:
    assert architecture_violations() == []


def test_backend_implementations_do_not_import_each_other() -> None:
    violations: list[str] = []
    backend_root = SOURCE / "backends"
    for path in backend_root.rglob("*.py"):
        relative = path.relative_to(backend_root)
        if len(relative.parts) < 2:
            continue
        owner = relative.parts[0]
        for imported in _imports(path):
            prefix = "lowbit_comm.backends."
            if imported.startswith(prefix):
                target = imported.removeprefix(prefix).split(".", 1)[0]
                if target != owner:
                    violations.append(f"{relative.as_posix()}: {imported}")
    assert violations == []


def test_cuda_hot_paths_avoid_host_worker_pools_and_tensor_lists() -> None:
    cuda = SOURCE / "backends" / "cuda"
    offenders: list[str] = []
    for name in ("executors.py", "dynamic_all_gather.py"):
        text = (cuda / name).read_text(encoding="utf-8")
        for forbidden in ("ThreadPoolExecutor", ".tolist()"):
            if forbidden in text:
                offenders.append(f"{name}: {forbidden}")
    assert offenders == []
