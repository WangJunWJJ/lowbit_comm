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
