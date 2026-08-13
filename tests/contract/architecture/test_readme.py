from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_readme_is_utf8_chinese_without_mojibake() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "低比特通信库" in text
    assert "鏄" not in text
    assert "锛" not in text


def test_readme_uses_current_registry_contract() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'registry.register("cuda", CudaBackend())' in text
    assert "registry.register(CudaBackend())" not in text


def test_readme_documents_stable_cuda_build_entrypoint() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "build_cuda_extension" in text
    assert "lowbit_comm_cuda_ops" in text
    assert "PYTHONPATH" in text


def test_design_documents_stable_cuda_build_entrypoint() -> None:
    text = (ROOT / "docs/SOFTWARE_DESIGN_ZH.md").read_text(encoding="utf-8")

    assert "build_cuda_extension" in text
    assert "lowbit_comm_cuda_ops" in text
