import json

import pytest

from tests.benchmarks.run_e2e_overlap_gate import (
    GateFailure,
    GateThresholds,
    evaluate_runs,
    main,
)


def candidate(
    mode: str,
    throughput: float,
    *,
    rank_parameters_consistent: bool = True,
    finite_loss: bool = True,
    final_loss: float = 1.0,
    initial_loss: float = 1.1,
    overlap_classification: str = "synchronous",
    overlap_efficiency: float = 0.0,
    world_size: int = 2,
    fallback_reason=None,
    effective_strategy: str = "all_gather",
) -> dict:
    return {
        "schema_version": 2,
        "mode": mode,
        "workload": {
            "synthetic": True,
            "data_root": None,
            "steps": 12,
            "warmup_steps": 2,
            "batch_size_per_rank": 16,
            "input_dim": 32,
            "hidden_dim": 64,
            "depth": 2,
            "num_classes": 8,
            "learning_rate": 0.001,
            "seed": 20260805,
            "device": "cuda",
            "dtype": "fp16",
            "bit": 8,
            "group_size": 64,
            "error_feedback": True,
            "bucket_cap_mb": 25,
        },
        "world_size": world_size,
        "global_batch_size": 32,
        "parameter_count": 1_000,
        "correctness": {
            "rank_parameters_consistent": rank_parameters_consistent,
            "max_parameter_difference": 0.0,
            "finite_loss": finite_loss,
        },
        "loss": {"initial": initial_loss, "final": final_loss},
        "timing": {
            "measured_steps": 10,
            "throughput_samples_per_second": throughput,
            "overlap_classification": overlap_classification,
            "overlap_efficiency": overlap_efficiency,
            "communication_ms": 2.0 if mode != "native_ddp" else 0.0,
            "compute_ms": 4.0,
            "overlapped_ms": 5.0 if mode == "ccdl_async" else 6.0,
            "exposed_communication_ms": 1.0 if mode == "ccdl_async" else 2.0,
        },
        "execution": {
            "fallback_reason": fallback_reason,
            "effective_strategy": effective_strategy,
            "capability": "cuda_extension",
        },
    }


def test_gate_rejects_speedup_with_rank_mismatch() -> None:
    with pytest.raises(GateFailure, match="rank parameters"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 110.0, rank_parameters_consistent=False),
            candidate(
                "ccdl_async",
                120.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_requires_async_to_beat_sync_compression() -> None:
    with pytest.raises(GateFailure, match="overlap benefit"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0),
            candidate(
                "ccdl_async",
                100.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_requires_async_to_beat_native_ddp_after_sync() -> None:
    with pytest.raises(GateFailure, match="native DDP"):
        evaluate_runs(
            candidate("native_ddp", 105.0),
            candidate("ccdl_sync", 100.0),
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_rejects_non_comparable_world_size() -> None:
    with pytest.raises(GateFailure, match="world_size"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0, world_size=4),
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_rejects_non_comparable_workload() -> None:
    async_run = candidate(
        "ccdl_async",
        102.0,
        overlap_classification="timeline_overlapped",
        overlap_efficiency=0.5,
    )
    async_run["workload"]["seed"] += 1
    with pytest.raises(GateFailure, match="workload"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0),
            async_run,
        )


def test_gate_rejects_incomplete_workload_signature() -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        run["workload"] = {"seed": 20260805}

    with pytest.raises(GateFailure, match="workload missing required fields"):
        evaluate_runs(*runs)


def test_gate_rejects_wrong_workload_field_type() -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        run["workload"]["seed"] = "20260805"

    with pytest.raises(GateFailure, match="workload.seed"):
        evaluate_runs(*runs)


@pytest.mark.parametrize(
    ("field", "value"),
    [("learning_rate", "0.001"), ("bit", 8.0)],
)
def test_gate_rejects_coercible_but_wrong_workload_types(field, value) -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        run["workload"][field] = value

    with pytest.raises(GateFailure, match=f"workload.{field}"):
        evaluate_runs(*runs)


@pytest.mark.parametrize(
    ("synthetic", "data_root"),
    [(True, "/dataset"), (False, None), (False, "")],
)
def test_gate_rejects_inconsistent_dataset_selection(synthetic, data_root) -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        run["workload"].update(synthetic=synthetic, data_root=data_root)

    with pytest.raises(GateFailure, match="dataset selection"):
        evaluate_runs(*runs)


@pytest.mark.parametrize("field", ["measured_steps", "global_batch_size"])
def test_gate_rejects_internally_inconsistent_run_dimensions(field) -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        if field == "measured_steps":
            run["timing"][field] += 1
        else:
            run[field] += 1

    with pytest.raises(GateFailure, match=field):
        evaluate_runs(*runs)


def test_gate_rejects_legacy_schema_even_when_all_runs_match() -> None:
    runs = [
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            102.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    ]
    for run in runs:
        run["schema_version"] = 1

    with pytest.raises(GateFailure, match="schema_version must be 2"):
        evaluate_runs(*runs)


@pytest.mark.parametrize("invalid_world_size", [True, "2", 2.5])
def test_gate_rejects_non_integer_comparability_fields(invalid_world_size) -> None:
    sync_run = candidate("ccdl_sync", 101.0)
    sync_run["world_size"] = invalid_world_size
    with pytest.raises(GateFailure, match="world_size must be a positive integer"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            sync_run,
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_requires_timeline_evidence_before_async_performance() -> None:
    with pytest.raises(GateFailure, match="timeline evidence"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0),
            candidate("ccdl_async", 102.0, overlap_classification="not_overlapped"),
        )


def test_gate_rejects_inconsistent_timeline_fields() -> None:
    async_run = candidate(
        "ccdl_async",
        102.0,
        overlap_classification="timeline_overlapped",
        overlap_efficiency=0.5,
    )
    async_run["timing"]["overlapped_ms"] = 99.0
    with pytest.raises(GateFailure, match="timeline union"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0),
            async_run,
        )


def test_gate_rejects_empty_fallback_marker() -> None:
    with pytest.raises(GateFailure, match="fallback_reason must be null"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0, fallback_reason=""),
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_rejects_non_cuda_extension_capability() -> None:
    sync_run = candidate("ccdl_sync", 101.0)
    sync_run["execution"]["capability"] = "torch_fallback"
    with pytest.raises(GateFailure, match="CUDA extension capability"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            sync_run,
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_rejects_native_strategy_disguised_as_compressed() -> None:
    with pytest.raises(GateFailure, match="compressed strategy"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0, effective_strategy="native_nccl"),
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


def test_gate_requires_loss_to_decrease() -> None:
    with pytest.raises(GateFailure, match="did not decrease"):
        evaluate_runs(
            candidate("native_ddp", 100.0, initial_loss=1.0, final_loss=1.1),
            candidate("ccdl_sync", 101.0, initial_loss=1.0, final_loss=1.1),
            candidate(
                "ccdl_async",
                102.0,
                initial_loss=1.0,
                final_loss=1.1,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


@pytest.mark.parametrize("field", ["finite_loss", "rank_parameters_consistent"])
def test_gate_requires_strict_boolean_correctness_fields(field) -> None:
    sync_run = candidate("ccdl_sync", 101.0)
    sync_run["correctness"][field] = "false"
    with pytest.raises(GateFailure, match=field):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            sync_run,
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


@pytest.mark.parametrize("difference", [float("nan"), -1.0, 0.1])
def test_gate_rejects_invalid_or_inconsistent_parameter_difference(difference) -> None:
    sync_run = candidate("ccdl_sync", 101.0)
    sync_run["correctness"]["max_parameter_difference"] = difference
    with pytest.raises(GateFailure, match="max_parameter_difference"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            sync_run,
            candidate(
                "ccdl_async",
                102.0,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_relative_loss_difference",
        "max_gradient_relative_l2",
        "min_async_sync_speedup",
        "min_async_native_speedup",
    ],
)
def test_gate_rejects_nan_thresholds(field) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        GateThresholds(**{field: float("nan")})


def test_gate_checks_loss_before_performance() -> None:
    with pytest.raises(GateFailure, match="loss divergence"):
        evaluate_runs(
            candidate("native_ddp", 100.0),
            candidate("ccdl_sync", 101.0, initial_loss=1.3, final_loss=1.2),
            candidate(
                "ccdl_async",
                103.0,
                initial_loss=1.3,
                final_loss=1.2,
                overlap_classification="timeline_overlapped",
                overlap_efficiency=0.5,
            ),
            thresholds=GateThresholds(max_relative_loss_difference=0.05),
        )


def test_passing_gate_reports_both_speedups() -> None:
    report = evaluate_runs(
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 105.0),
        candidate(
            "ccdl_async",
            110.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    )

    assert report["passed"] is True
    assert report["speedup"]["async_vs_sync"] == pytest.approx(110 / 105)
    assert report["speedup"]["async_vs_native"] == 1.1


def test_cli_records_non_passing_gate_without_hiding_results(tmp_path) -> None:
    paths = []
    runs = (
        candidate("native_ddp", 100.0),
        candidate("ccdl_sync", 101.0),
        candidate(
            "ccdl_async",
            100.0,
            overlap_classification="timeline_overlapped",
            overlap_efficiency=0.5,
        ),
    )
    for index, run in enumerate(runs):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "gate.json"

    exit_code = main(
        [
            "--native",
            str(paths[0]),
            "--sync",
            str(paths[1]),
            "--async",
            str(paths[2]),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["passed"] is False
    assert "overlap benefit" in report["failure"]
    assert report["runs"]["ccdl_async"]["timing"][
        "throughput_samples_per_second"
    ] == 100.0


def test_cli_records_malformed_input_failure(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(candidate("ccdl_sync", 101.0)), encoding="utf-8")
    output = tmp_path / "gate.json"

    exit_code = main(
        [
            "--native",
            str(malformed),
            "--sync",
            str(valid),
            "--async",
            str(valid),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["failure_stage"] == "input"
    assert "JSONDecodeError" in report["failure"]
