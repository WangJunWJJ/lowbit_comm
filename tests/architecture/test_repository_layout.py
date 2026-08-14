from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_v040_uses_root_package_and_root_csrc() -> None:
    assert (ROOT / "lowbit_comm" / "__init__.py").is_file()
    assert not (ROOT / "src" / "lowbit_comm").exists()
    assert (ROOT / "csrc").is_dir()


def test_v030_python_tree_is_not_active() -> None:
    assert not (ROOT / "src").exists()
    assert not (ROOT / "docs" / "architecture").exists()
    assert not (ROOT / "tests" / "v03").exists()


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
    assert 'version = "0.4.0.dev0"' in pyproject
    assert 'where = ["."]' in pyproject
