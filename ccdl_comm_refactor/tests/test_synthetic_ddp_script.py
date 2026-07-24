from pathlib import Path


def test_synthetic_ddp_compare_exposes_bucket_gate_and_model_size_args() -> None:
    source = (Path(__file__).parent / "distributed" / "synthetic_ddp_compare.py").read_text(encoding="utf-8")

    assert "--min-compress-numel" in source
    assert "--bucket-cap-mb" in source
    assert "--width" in source
    assert "create_ddp_comm_hook" in source
    assert "parameter_count" in source
