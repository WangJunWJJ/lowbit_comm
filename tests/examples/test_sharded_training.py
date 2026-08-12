from __future__ import annotations

from argparse import Namespace

import pytest

from examples.sharded_training import MODES, config_from_args
from examples.sharded_training import ShardedRunConfig, run
from examples.training.config import TrainingConfig
from examples.training.sharded_metrics import (
    PHASE_NAMES,
    ShardedPhaseMetrics,
    augment_training_payload,
)


def test_config_maps_public_modes_without_changing_workload() -> None:
    arguments = Namespace(
        config=None,
        mode="native_ddp",
        synthetic=True,
        data_root=None,
        steps=3,
        warmup_steps=1,
        batch_size_per_rank=2,
        input_dim=32,
        hidden_dim=64,
        depth=2,
        num_classes=8,
        learning_rate=None,
        seed=None,
        device="cpu",
        dtype="fp32",
        bit=None,
        group_size=None,
        bucket_cap_mb=None,
        error_feedback=None,
        output=None,
    )
    native = config_from_args(arguments)
    arguments.mode = "ccdl_sharded_sgd"
    sharded = config_from_args(arguments)

    assert MODES == ("native_ddp", "ccdl_full_gradient", "ccdl_sharded_sgd")
    assert native.mode == "native_ddp"
    assert sharded.mode == "ccdl_sharded_sgd"
    assert native.training.comparison_workload() == sharded.training.comparison_workload()


def test_phase_metrics_require_complete_finite_step_samples() -> None:
    samples = {name: (1.0, 2.0) for name in PHASE_NAMES}
    metrics = ShardedPhaseMetrics(measured_steps=2, samples_ms=samples)

    assert set(metrics.to_dict()) == set(PHASE_NAMES)
    assert metrics.to_dict()["compressed_reduce_scatter"] == 1.5

    samples.pop("parameter_writeback")
    with pytest.raises(ValueError, match="phase names"):
        ShardedPhaseMetrics(measured_steps=2, samples_ms=samples)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_phase_metrics_reject_invalid_values(invalid) -> None:
    samples = {name: (1.0,) for name in PHASE_NAMES}
    samples["local_shard_update"] = (invalid,)

    with pytest.raises(ValueError, match="finite and >= 0"):
        ShardedPhaseMetrics(measured_steps=1, samples_ms=samples)


def test_augment_payload_records_schema_phases_and_pointer_reuse() -> None:
    payload = {
        "schema_version": 2,
        "mode": "native_ddp",
        "workload": {"seed": 1},
        "execution": {"requested_mode": "native_ddp"},
    }
    phases = ShardedPhaseMetrics(
        measured_steps=1,
        samples_ms={name: (0.0,) for name in PHASE_NAMES},
    )

    result = augment_training_payload(
        payload,
        mode="ccdl_sharded_sgd",
        phases=phases,
        phases_measured=True,
        initial_pointers={"flat_gradients": 1, "reduced_gradient": 2},
        final_pointers={"flat_gradients": 1, "reduced_gradient": 2},
    )

    assert result["schema_version"] == 3
    assert result["mode"] == "ccdl_sharded_sgd"
    assert result["execution"]["requested_mode"] == "ccdl_sharded_sgd"
    assert result["phase_timing_ms"] == {name: 0.0 for name in PHASE_NAMES}
    assert result["phase_timing_measured"] is True
    assert result["buffer_reuse"]["stable"] is True


def test_augment_payload_rejects_pointer_changes() -> None:
    phases = ShardedPhaseMetrics(
        measured_steps=1,
        samples_ms={name: (0.0,) for name in PHASE_NAMES},
    )

    with pytest.raises(ValueError, match="buffer pointers changed"):
        augment_training_payload(
            {"execution": {}},
            mode="ccdl_sharded_sgd",
            phases=phases,
            phases_measured=True,
            initial_pointers={"flat_gradients": 1},
            final_pointers={"flat_gradients": 2},
        )


def test_single_rank_cpu_sharded_training_smoke() -> None:
    pytest.importorskip("torch")
    payload = run(
        ShardedRunConfig(
            mode="ccdl_sharded_sgd",
            training=TrainingConfig(
                mode="native_ddp",
                synthetic=True,
                steps=2,
                warmup_steps=1,
                batch_size_per_rank=2,
                input_dim=8,
                hidden_dim=16,
                depth=2,
                num_classes=4,
                device="cpu",
                dtype="fp32",
            ),
        )
    )

    assert payload is not None
    assert payload["mode"] == "ccdl_sharded_sgd"
    assert payload["schema_version"] == 3
    assert payload["correctness"]["rank_parameters_consistent"] is True
    assert payload["correctness"]["finite_loss"] is True
    assert payload["buffer_reuse"]["stable"] is True
    assert set(payload["phase_timing_ms"]) == set(PHASE_NAMES)
