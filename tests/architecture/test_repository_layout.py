import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_CORE_IMPORTS = (
    "lowbit_comm.backends",
    "lowbit_comm.compiler",
)


class RuntimeImportVisitor(ast.NodeVisitor):
    """Collect imports that Python can execute at runtime."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.imports: list[tuple[int, str]] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(
            (node.lineno, alias.name) for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve_from_module(
            self.package,
            node.level,
            node.module,
        )
        for alias in node.names:
            imported = module
            if alias.name != "*":
                imported = f"{module}.{alias.name}" if module else alias.name
            self.imports.append((node.lineno, imported))


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
    ) or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


def _runtime_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_parts = path.relative_to(ROOT).with_suffix("").parts
    package = ".".join(module_parts[:-1])
    visitor = RuntimeImportVisitor(package)
    visitor.visit(tree)
    return visitor.imports


def _resolve_from_module(
    package: str,
    level: int,
    module: str | None,
) -> str:
    if level == 0:
        return module or ""
    package_parts = package.split(".")
    parent_count = level - 1
    base = package_parts[: len(package_parts) - parent_count]
    if module is not None:
        base.extend(module.split("."))
    return ".".join(base)


def _is_forbidden_core_import(imported: str) -> bool:
    return any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_CORE_IMPORTS
    )


def test_v040_uses_root_package_and_root_csrc() -> None:
    assert (ROOT / "lowbit_comm" / "__init__.py").is_file()
    assert not (ROOT / "src" / "lowbit_comm").exists()
    assert (ROOT / "csrc").is_dir()


def test_v030_python_tree_is_not_active() -> None:
    assert not (ROOT / "src").exists()
    assert not (ROOT / "docs" / "architecture").exists()
    assert not (ROOT / "tests" / "v03").exists()


def test_core_runtime_imports_do_not_target_compiler_or_backends() -> None:
    violations = []
    core = ROOT / "lowbit_comm" / "core"
    for path in sorted(core.rglob("*.py")):
        for line, imported in _runtime_imports(path):
            if _is_forbidden_core_import(imported):
                relative = path.relative_to(ROOT).as_posix()
                violations.append(f"{relative}:{line}: {imported}")

    assert not violations, "forbidden Core runtime imports:\n" + "\n".join(
        violations
    )


def test_only_formal_documents_are_tracked() -> None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "docs"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(result.stdout.splitlines()) == {
        "docs/SOFTWARE_DESIGN_ZH.md",
        "docs/SOFTWARE_REQUIREMENTS_ZH.md",
    }


def test_development_version_is_v040() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = (ROOT / "lowbit_comm" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "lowbit_comm._version.__version__"}' in pyproject
    assert '__version__ = "0.4.0.dev0"' in version
    assert 'where = ["."]' in pyproject


def test_top_level_import_does_not_load_torch_or_cuda_extension() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import lowbit_comm; "
        "assert 'torch' not in sys.modules; "
        "assert 'lowbit_comm._C' not in sys.modules"
    )
    environment = {"PATH": os.environ["PATH"]}
    subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        env=environment,
    )
