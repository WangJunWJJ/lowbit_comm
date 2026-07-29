from pathlib import Path


def test_legacy_synthetic_mlp_script_uses_legacy_ccdl_gradient_sync() -> None:
    source = (Path(__file__).resolve().parent / "distributed" / "legacy_synthetic_mlp_compare.py").read_text(
        encoding="utf-8"
    )

    assert "from ccdl.comm import qall_reduce" in source
    assert "from ccdl.quantization import Quantizer" in source
    assert "SyntheticMLP" in source
    assert '"samples_per_s"' in source
    assert '"avg_step_ms"' in source
