from __future__ import annotations

import json

import pytest

from tests.benchmarks.qwd_parameter_pipeline_gate import GateFailure, evaluate
from tests.benchmarks.run_sharded_training_gate import evaluate_qwd_trials


MODES = ("native_ddp", "sharded_fp", "sharded_compressed", "sharded_qwd")


def write_trials(
    root,
    *,
    native: float = 353.0,
    sharded_fp: float = 350.0,
    direct_int8: float = 380.0,
    qwd: float = 375.0,
    fp_loss: float = 2.94,
    qwd_loss: float = 2.97,
    qwd_fallback: str | None = None,
    workspace_stable: bool = True,
) -> None:
    throughputs = {
        "native_ddp": native,
        "sharded_fp": sharded_fp,
        "sharded_compressed": direct_int8,
        "sharded_qwd": qwd,
    }
    losses = {
        "native_ddp": 2.98,
        "sharded_fp": fp_loss,
        "sharded_compressed": 4.70,
        "sharded_qwd": qwd_loss,
    }
    for mode in MODES:
        for trial in range(1, 4):
            payload = {
                "mode": mode,
                "world_size": 4,
                "timing": {
                    "throughput_samples_per_second": throughputs[mode]
                },
                "loss": {"final": losses[mode]},
                "correctness": {
                    "finite_loss": True,
                    "max_parameter_difference": 0.0,
                },
                "fallback_reason": (
                    qwd_fallback if mode == "sharded_qwd" else None
                ),
                "selected_fast_path": (
                    "fused_int8_qwd" if mode == "sharded_qwd" else mode
                ),
                "workspace_pointers": {
                    "stable": workspace_stable if mode == "sharded_qwd" else True
                },
                "parameter_communication": (
                    {
                        "algorithm": "qwd",
                        "bit": 8,
                        "decision_counts": {"qwd": 100, "fp_refresh": 1},
                        "sampled_relative_errors": [0.001, 0.002],
                    }
                    if mode == "sharded_qwd"
                    else None
                ),
                "benchmark_environment": {
                    "physical_gpu_ids": [1, 2, 3, 4],
                    "image_id": "sha256:image",
                    "source_hash": "abc123",
                    "extension_sha256": "def456",
                },
            }
            (root / f"{mode}-trial{trial}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )


def test_qwd_gate_requires_precision_and_direct_int8_performance(tmp_path) -> None:
    write_trials(tmp_path)

    result = evaluate(tmp_path, world_size=4)

    assert result["qwd_speedup_vs_native"] >= 1.05
    assert result["qwd_ratio_vs_direct_int8"] >= 0.98
    assert result["qwd_loss_ratio_vs_sharded_fp"] <= 1.02
    assert result["passed"] is True


def test_sharded_gate_entrypoint_dispatches_qwd_trials(tmp_path) -> None:
    write_trials(tmp_path)

    result = evaluate_qwd_trials(tmp_path, world_size=4)

    assert result["passed"] is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"qwd": 371.0}, "direct INT8"),
        ({"qwd_loss": 3.1}, "loss ratio"),
        ({"qwd_fallback": "fp refresh"}, "fallback"),
        ({"workspace_stable": False}, "workspace"),
    ),
)
def test_qwd_gate_rejects_invalid_evidence(tmp_path, overrides, message) -> None:
    write_trials(tmp_path, **overrides)

    with pytest.raises(GateFailure, match=message):
        evaluate(tmp_path, world_size=4)


def test_qwd_gate_requires_three_trials_per_mode(tmp_path) -> None:
    write_trials(tmp_path)
    (tmp_path / "sharded_qwd-trial3.json").unlink()

    with pytest.raises(GateFailure, match="three trials"):
        evaluate(tmp_path, world_size=4)
