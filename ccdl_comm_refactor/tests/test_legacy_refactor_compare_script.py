from pathlib import Path


def test_legacy_refactor_compare_script_uses_both_ccdl_apis() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "legacy_refactor_compare.py").read_text(
        encoding="utf-8"
    )

    assert "from ccdl.comm import qall_reduce" in source
    assert "from ccdl.quantization import Quantizer" in source
    assert "from ccdl_comm import CompressionConfig, compressed_all_reduce" in source
    assert '"legacy_ccdl_ms"' in source
    assert '"refactor_ccdl_ms"' in source
    assert '"speedup_refactor_over_legacy"' in source
