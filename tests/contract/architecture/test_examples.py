from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_ddp_training_example_exposes_fair_comparison_metrics() -> None:
    source = (ROOT / "examples/train_ddp.py").read_text(encoding="utf-8")

    for mode in (
        '"native"',
        '"compressed_all_gather"',
        '"compressed_rs_ag"',
    ):
        assert mode in source
    for metric in (
        '"samples_per_second"',
        '"step_p50_ms"',
        '"step_p95_ms"',
        '"loss_start"',
        '"loss_end"',
        '"rank_weight_gap"',
    ):
        assert metric in source
    assert "DistributedDataParallel" in source
    assert "register_comm_hook" in source
    assert 'hook.__annotations__ = {' in source
    assert '"bucket": dist.GradBucket' in source
