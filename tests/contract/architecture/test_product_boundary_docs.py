"""Formal documentation contracts for the v0.4.0 product boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_user_docs_keep_rsag_experimental_and_native_fail_closed() -> None:
    readme = _read("README.md")

    assert "lowbit_comm.experimental" in readme
    assert "Native 默认" in readme
    assert "所有 seed 收益" in readme
    assert "严格大于 0" in readme
    assert "CAG 训练：BLOCKED" in readme


def test_formal_docs_define_the_exact_runtime_matrix_and_async_boundary() -> None:
    requirements = _read("docs/SOFTWARE_REQUIREMENTS_ZH.md")
    design = _read("docs/SOFTWARE_DESIGN_ZH.md")

    for source in (requirements, design):
        assert "2.5.0a0+872d972e41.nv24.08" in source
        assert "CUDA 12.6" in source
        assert "NCCL 2.22.3" in source
        assert "扩展 ABI 1" in source
        assert "transport 仍同步等待" in source
        assert "FullTensorResult" in source
        assert "RSAG/qWD" in source


def test_changelog_records_production_hardening_without_auto_claim() -> None:
    changelog = _read("CHANGELOG.md")

    assert "Production hardening" in changelog
    assert "experimental RSAG/qWD" in changelog
    assert "CAG remains blocked" in changelog
