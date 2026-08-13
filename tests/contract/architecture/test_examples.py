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
        '"schema_version"',
        '"fingerprint"',
        '"rounds"',
        '"samples_ms"',
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
    assert "class ResidualBlock" in source
    assert "value + self.scale *" in source
    assert "targets = inputs.mul(0.5)" in source
    assert 'hook.__annotations__ = {' in source
    assert '"bucket": dist.GradBucket' in source


def test_cifar10_example_exposes_reproducible_quality_metrics() -> None:
    source = (ROOT / "examples/train_cifar10.py").read_text(encoding="utf-8")

    for public_component in (
        "datasets.CIFAR10",
        "models.resnet18",
        "DistributedSampler",
        "set_epoch",
    ):
        assert public_component in source
    for mode in (
        '"native"',
        '"compressed_all_gather"',
        '"compressed_rs_ag"',
    ):
        assert mode in source
    for metric in (
        "schema_version",
        "fingerprint",
        "epoch_samples_per_second",
        "validation_accuracy",
        "validation_loss",
        "time_to_target_seconds",
        "rank_weight_gap",
    ):
        assert metric in source
    assert "DistributedDataParallel" in source
    assert "register_comm_hook" in source
    assert "GradScaler" in source
    assert 'choices=("cifar10", "fake")' in source
    assert "datasets.FakeData" in source
