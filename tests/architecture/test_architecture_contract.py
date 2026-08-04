from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]
CONTRACT = ROOT / "docs" / "architecture" / "architecture_contract.json"
PACKAGE = ROOT / "ccdl_comm"


def load_architecture_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _matches_forbidden(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(f"{forbidden}.")


def test_architecture_contract_declares_control_data_boundary() -> None:
    contract = load_architecture_contract()

    assert contract["version"] == 1
    assert contract["control_plane_boundary"] == "CompiledCommunicationPlan"
    assert contract["explicit_strategy_default"] == "strict"
    assert contract["auto_strategy_opt_in"] is True
    assert "registry_lookup" in contract["hot_path_forbidden"]
    assert "capability_probe" in contract["hot_path_forbidden"]
    assert "process_group_creation" in contract["hot_path_forbidden"]


def test_declared_core_modules_do_not_import_forbidden_backends() -> None:
    contract = load_architecture_contract()
    forbidden = tuple(contract["core_forbidden_imports"])

    violations: list[str] = []
    for module in contract["core_modules"]:
        path = PACKAGE / f"{module}.py"
        if not path.exists():
            continue
        for imported in _imports(path):
            if any(_matches_forbidden(imported, item) for item in forbidden):
                violations.append(f"{module}: {imported}")

    assert violations == []


def test_backend_packages_do_not_cross_import() -> None:
    contract = load_architecture_contract()
    assert contract["backend_cross_imports_forbidden"] is True

    backend_forbidden = {
        "cuda": ("ccdl_comm.ascend",),
        "ascend": ("ccdl_comm.cuda",),
    }
    violations: list[str] = []
    for backend, forbidden in backend_forbidden.items():
        root = PACKAGE / backend
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            for imported in _imports(path):
                if any(_matches_forbidden(imported, item) for item in forbidden):
                    violations.append(f"{path.relative_to(PACKAGE)}: {imported}")

    assert violations == []
