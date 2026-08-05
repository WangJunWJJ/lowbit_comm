from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tests.benchmarks.run_sharded_training_gate import (
    GateFailure,
    ShardedGateThresholds,
    evaluate_sharded_runs,
    main,
)


def _candidate(mode: str, throughput: float, *, world_size: int = 2) -> dict:
    workload = {
        "synthetic": True,
        "data_root": None,
        "steps": 22,
        "warmup_steps": 2,
        "batch_size_per_rank": 16,
        "input_dim": 1024,
        "hidden_dim": 4096,
        "depth": 4,
        "num_classes": 1024,
        "learning_rate": 0.001,
        "seed": 20260805,
        "device": "cuda",
        "dtype": "fp16",
        "bit": 8,
        "group_size": 64,
        "error_feedback": True,
        "bucket_cap_mb": 25,
    }
    sharded = mode == "ccdl_sharded_sgd"
    full = mode == "ccdl_full_gradient"
    pointers = {
        "flat_gradients": 1,
        "reduced_gradient": 2,
        "local_parameters": 3,
        "gathered_parameters": 4,
    }
    return {
        "schema_version": 3,
        "mode": mode,
        "workload": workload,
        "world_size": world_size,
        "global_batch_size": 16 * world_size,
        "parameter_count": 44_971_744,
        "timing": {
            "measured_steps": 20,
            "throughput_samples_per_second": throughput,
            "mean_step_latency_ms": 10.0,
        },
        "loss": {"initial": 2.0, "final": 1.0, "samples": [2.0, 1.0]},
        "correctness": {
            "finite_loss": True,
            "rank_parameters_consistent": True,
            "max_parameter_difference": 0.0,
        },
        "execution": {
            "requested_mode": mode,
            "effective_strategy": (
                "compressed" if sharded else "all_gather" if full else "native"
            ),
            "capability": "cuda_extension" if (sharded or full) else "native",
            "fallback_reason": None,
            **({"fast_path": "cuda_reduced_shard", "output_layout": "shard"} if sharded else {}),
        },
        "phase_timing_ms": {
            "backward_and_flatten": 4.0 if sharded else 0.0,
            "compressed_reduce_scatter": 2.0 if sharded else 0.0,
            "local_shard_update": 1.0 if sharded else 0.0,
            "parameter_all_gather": 1.0 if sharded else 0.0,
            "parameter_writeback": 1.0 if sharded else 0.0,
        },
        "phase_timing_measured": sharded,
        "buffer_reuse": {
            "stable": True,
            "initial_pointers": pointers if sharded else {},
            "final_pointers": pointers if sharded else {},
        },
    }


def _valid_runs(*, world_size: int = 2):
    return (
        _candidate("native_ddp", 100.0, world_size=world_size),
        _candidate("ccdl_full_gradient", 90.0, world_size=world_size),
        _candidate("ccdl_sharded_sgd", 96.0 if world_size == 2 else 91.0, world_size=world_size),
    )


def test_two_gpu_gate_accepts_native_ratio_at_floor() -> None:
    native, full, sharded = _valid_runs()
    sharded["timing"]["throughput_samples_per_second"] = 95.0

    result = evaluate_sharded_runs(native, full, sharded)

    assert result["passed"] is True
    assert result["speedup"]["sharded_vs_native"] == pytest.approx(0.95)


def test_four_gpu_gate_requires_sharded_consumer_benefit() -> None:
    native, full, sharded = _valid_runs(world_size=4)
    sharded["timing"]["throughput_samples_per_second"] = 90.0

    with pytest.raises(GateFailure, match="sharded consumer benefit"):
        evaluate_sharded_runs(native, full, sharded)


@pytest.mark.parametrize(
    ("mutation", "stage"),
    [
        (lambda runs: runs[2].update(world_size=4), "comparability"),
        (lambda runs: runs[2].update(global_batch_size=31), "comparability"),
        (lambda runs: runs[2]["workload"].update(seed=7), "comparability"),
        (lambda runs: runs[2]["correctness"].update(finite_loss="true"), "correctness"),
        (lambda runs: runs[2]["correctness"].update(rank_parameters_consistent="true"), "correctness"),
        (lambda runs: runs[2]["correctness"].update(max_parameter_difference=1e-6), "correctness"),
        (lambda runs: runs[2]["execution"].update(fallback_reason="fallback"), "execution"),
        (lambda runs: runs[2]["execution"].update(capability="torch_fallback"), "execution"),
        (lambda runs: runs[2]["buffer_reuse"].update(stable="true"), "execution"),
        (lambda runs: runs[2]["loss"].update(final=float("nan")), "convergence"),
    ],
)
def test_gate_rejects_invalid_or_bypassed_evidence(mutation, stage) -> None:
    runs = [deepcopy(run) for run in _valid_runs()]
    mutation(runs)

    with pytest.raises(GateFailure) as failure:
        evaluate_sharded_runs(*runs)

    assert failure.value.stage == stage


def test_gate_rejects_loss_difference_above_two_percent() -> None:
    native, full, sharded = _valid_runs()
    sharded["loss"]["final"] = 1.021

    with pytest.raises(GateFailure, match="relative final loss") as failure:
        evaluate_sharded_runs(native, full, sharded)

    assert failure.value.stage == "convergence"


def test_gate_rejects_two_gpu_ratio_below_floor() -> None:
    native, full, sharded = _valid_runs()
    sharded["timing"]["throughput_samples_per_second"] = 94.9

    with pytest.raises(GateFailure, match="native DDP ratio") as failure:
        evaluate_sharded_runs(native, full, sharded)

    assert failure.value.stage == "performance"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_relative_loss_difference": float("nan")},
        {"max_relative_loss_difference": 0.021},
        {"min_2gpu_native_ratio": 0.949},
        {"min_4gpu_full_gradient_ratio": 0.999},
    ],
)
def test_thresholds_disallow_nonfinite_or_weaker_safety_floors(kwargs) -> None:
    with pytest.raises(ValueError):
        ShardedGateThresholds(**kwargs)


def test_cli_persists_input_failure_report_for_malformed_json(tmp_path) -> None:
    native = tmp_path / "native.json"
    full = tmp_path / "full.json"
    sharded = tmp_path / "sharded.json"
    report = tmp_path / "report.json"
    native.write_text("{bad", encoding="utf-8")
    full.write_text(json.dumps(_valid_runs()[1]), encoding="utf-8")
    sharded.write_text(json.dumps(_valid_runs()[2]), encoding="utf-8")

    code = main(
        [
            "--native", str(native),
            "--full-gradient", str(full),
            "--sharded", str(sharded),
            "--output", str(report),
        ]
    )

    assert code == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["failure_stage"] == "input"
    assert "native" in payload["raw_inputs"]
