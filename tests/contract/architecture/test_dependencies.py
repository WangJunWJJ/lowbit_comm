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
        for imported in _imports(path):
            if any(_matches(imported, item) for item in forbidden):
                violations.append(f"{relative.as_posix()}: {imported}")
    return sorted(set(violations))


def test_new_source_never_imports_legacy_control_plane() -> None:
    assert architecture_violations() == []
