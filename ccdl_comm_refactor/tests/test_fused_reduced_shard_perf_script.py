from __future__ import annotations

import sys

import pytest

from tests.benchmarks.fused_reduced_shard_gate import evaluate
from tests.distributed.fused_reduced_shard_perf import (
    _abba_positions,
    _baseline_extension_status,
    _fused_kernel_launch_count,
    _profiler_evidence,
    parse_args,
)
from ccdl_comm.cuda.loader import CudaExtensionStatus


_REQUIRED_CASES = tuple(
    (world_size, bucket_mib, mode)
    for world_size in (2, 4)
    for bucket_mib in (1, 16, 64)
    for mode in ("caller", "lease")
)


def _result(
    world_size: int, bucket_mib: int, mode: str, **overrides: object
) -> dict[str, object]:
    task12_ms = 1.0 + world_size + bucket_mib / 100.0
    result: dict[str, object] = {
        "world_size": world_size,
        "bucket_mib": bucket_mib,
        "dtype": "fp16",
        "bit": 8,
        "group_size": 64,
        "output_mode": mode,
        "warmup": 20,
        "repeat": 100,
        "measurement_order": "task12-fused-fused-task12",
        "task12_ms": task12_ms,
        "fused_ms": task12_ms * 0.95,
        "speedup": 1.0 / 0.95,
        "task12_peak_memory_bytes": 4096,
        "fused_peak_memory_bytes": 2048,
        "steady_allocation_bytes": 0,
        "relative_l2": 0.005,
        "max_abs_error": 0.01,
        "non_finite": 0,
        "fused_kernel_launches": 1,
        "fallback_used": False,
        "fallback_reason": None,
        "output_pointer_stable": True,
        "output_pointers": [101, 101],
        "per_position_samples_ms": {
            "task12_first": [task12_ms],
            "fused_first": [task12_ms * 0.95],
            "fused_second": [task12_ms * 0.95],
            "task12_second": [task12_ms],
        },
        "per_position_medians_ms": {
            "task12_first": task12_ms,
            "fused_first": task12_ms * 0.95,
            "fused_second": task12_ms * 0.95,
            "task12_second": task12_ms,
        },
        "fused_metadata": {
            "fused_dequant_reduce": True,
            "output_ownership": "caller",
        },
        "profiler": {
            "production_fused_kernel_names": ["dequant_reduce_fused_16bit_kernel"],
            "production_fused_kernel_launches": 1,
            "kernel_names": ["dequant_reduce_fused_16bit_kernel"],
        },
        "allocation_evidence": {
            "allocated_before_bytes": 8192,
            "candidate_peak_bytes": 8192,
            "allocated_after_bytes": 8192,
        },
        "identity": {"commit": "abcdef0", "hostname": "a6000"},
        "no_full_gradient_restoration": True,
        "seed": 20260802,
        "reference": "fp16_all_reduce",
        "run_id": "placeholder",
        "started_at": "2026-08-02T00:00:00+00:00",
        "rank_evidence": [
            {
                "rank": rank,
                "relative_l2": 0.005,
                "non_finite": 0,
                "fused_kernel_launches": 1,
                "profiler": {
                    "kernel_names": ["dequant_reduce_fused_16bit_kernel"],
                    "production_fused_kernel_names": [
                        "dequant_reduce_fused_16bit_kernel"
                    ],
                    "production_fused_kernel_launches": 1,
                },
                "fallback_used": False,
                "output_pointer_stable": True,
                "output_pointers": [101, 101],
                "fused_metadata": {
                    "fused_dequant_reduce": True,
                    "output_ownership": "caller",
                },
                "allocation_evidence": {
                    "allocated_before_bytes": 8192,
                    "candidate_peak_bytes": 8192,
                    "allocated_after_bytes": 8192,
                },
            }
            for rank in range(world_size)
        ],
    }
    result.update(overrides)
    return result


def _complete_results() -> list[dict[str, object]]:
    return [
        _result(
            world_size,
            bucket_mib,
            mode,
            run_id=f"{world_size}-{bucket_mib}-{mode}-{run}",
        )
        for world_size, bucket_mib, mode in _REQUIRED_CASES
        for run in range(5)
    ]


def test_parse_args_exposes_task12_1_matrix_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fused_reduced_shard_perf.py",
            "--bucket-mib=16",
            "--dtype=fp16",
            "--bit=8",
            "--group-size=64",
            "--mode=lease",
            "--warmup=20",
            "--repeat=100",
            "--output-json=/tmp/result.json",
        ],
    )

    args = parse_args()

    assert args.bucket_mib == 16
    assert args.dtype == "fp16"
    assert args.bit == 8
    assert args.group_size == 64
    assert args.mode == "lease"
    assert args.warmup == 20
    assert args.repeat == 100
    assert args.output_json.name == "result.json"


def test_abba_loop_preserves_all_four_position_keys() -> None:
    observed: dict[str, str] = {}

    for key, operation_name in _abba_positions():
        observed[key] = operation_name

    assert observed == {
        "task12_first": "task12",
        "fused_first": "fused",
        "fused_second": "fused",
        "task12_second": "task12",
    }


def test_profiler_counts_kernel_launches_not_distinct_kernel_names() -> None:
    class Event:
        def __init__(self, key: str, count: int) -> None:
            self.key = key
            self.count = count

    assert (
        _fused_kernel_launch_count(
            [
                Event("dequant_reduce_fused_16bit_kernel", 2),
                Event("unrelated_kernel", 99),
            ]
        )
        == 2
    )


def test_profiler_evidence_preserves_all_observed_kernel_names() -> None:
    class Event:
        def __init__(self, key: str, count: int) -> None:
            self.key = key
            self.count = count

    evidence = _profiler_evidence(
        [
            Event("dequant_reduce_fused_16bit_kernel", 1),
            Event("quantize_kernel", 4),
        ]
    )

    assert evidence == {
        "kernel_names": ["dequant_reduce_fused_16bit_kernel", "quantize_kernel"],
        "production_fused_kernel_names": ["dequant_reduce_fused_16bit_kernel"],
        "production_fused_kernel_launches": 1,
    }


def test_baseline_extension_hides_only_the_fused_mean_symbol() -> None:
    class Extension:
        inplace_dequantize_reduce_mean = object()
        another_symbol = object()

    status = _baseline_extension_status(CudaExtensionStatus(True, Extension()))

    assert status.available is True
    assert status.module is not None
    assert not hasattr(status.module, "inplace_dequantize_reduce_mean")
    assert status.module.another_symbol is Extension.another_symbol


def test_gate_accepts_complete_caller_and_lease_matrix() -> None:
    assert evaluate(_complete_results()) == []


def test_gate_requires_exactly_five_runs_for_every_mode() -> None:
    results = _complete_results()
    results.pop()

    failures = evaluate(results)

    assert any(
        "4gpu/64MiB/lease: requires exactly 5 runs; received 4" in failure
        for failure in failures
    )


def test_gate_requires_unique_run_id_per_case() -> None:
    results = _complete_results()
    results[1]["run_id"] = results[0]["run_id"]

    failures = evaluate(results)

    assert any("requires 5 unique run_id values" in failure for failure in failures)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"fallback_used": True}, "production fused path used fallback"),
        ({"fused_kernel_launches": 2}, "expected one fused dequant-reduce-mean launch"),
        ({"steady_allocation_bytes": 64}, "steady-state allocation is non-zero"),
        ({"relative_l2": 0.021}, "accuracy gate failed"),
        ({"non_finite": 1}, "accuracy gate failed"),
        ({"output_pointer_stable": False}, "output pointer is unstable"),
        ({"measurement_order": "task12-fused"}, "unbalanced measurement order"),
        ({"dtype": "bf16"}, "requires dtype='fp16'"),
        ({"bit": 4}, "requires bit=8"),
        ({"group_size": 32}, "requires group_size=64"),
        ({"warmup": 10}, "requires warmup=20"),
        ({"repeat": 30}, "requires repeat=100"),
        ({"seed": 1}, "requires seed=20260802"),
    ],
)
def test_gate_rejects_invalid_production_evidence(
    overrides: dict[str, object], expected: str
) -> None:
    results = _complete_results()
    results[0].update(overrides)

    failures = evaluate(results)

    assert any(expected in failure for failure in failures)


def test_gate_rejects_rank_level_allocation_and_accuracy_failures() -> None:
    results = _complete_results()
    evidence = results[0]["rank_evidence"]
    assert isinstance(evidence, list)
    evidence[0]["non_finite"] = 1
    evidence[0]["allocation_evidence"] = {
        "allocated_before_bytes": 8192,
        "candidate_peak_bytes": 8256,
        "allocated_after_bytes": 8192,
    }

    failures = evaluate(results)

    assert any("rank 0: accuracy gate failed" in failure for failure in failures)
    assert any(
        "rank 0: steady-state allocation is non-zero" in failure for failure in failures
    )


def test_gate_rejects_duplicate_or_missing_rank_evidence() -> None:
    results = _complete_results()
    evidence = results[0]["rank_evidence"]
    assert isinstance(evidence, list)
    evidence[1]["rank"] = 0

    failures = evaluate(results)

    assert any(
        "rank evidence must contain each rank exactly once" in failure
        for failure in failures
    )


def test_gate_rejects_forged_profiler_names_and_inconsistent_counts() -> None:
    results = _complete_results()
    results[0]["profiler"] = {
        "kernel_names": ["ordinary_kernel"],
        "production_fused_kernel_names": ["not_actually_fused"],
        "production_fused_kernel_launches": 1,
    }
    evidence = results[0]["rank_evidence"]
    assert isinstance(evidence, list)
    evidence[0]["profiler"] = {
        "kernel_names": ["dequant_reduce_fused_16bit_kernel"],
        "production_fused_kernel_names": ["dequant_reduce_fused_16bit_kernel"],
        "production_fused_kernel_launches": 2,
    }

    failures = evaluate(results)

    assert any(
        "profiler kernel names do not prove fused execution" in failure
        for failure in failures
    )
    assert any(
        "rank 0: profiler launch count is inconsistent" in failure
        for failure in failures
    )


def test_gate_rejects_forged_abba_samples_and_medians() -> None:
    results = _complete_results()
    results[0]["per_position_samples_ms"] = {
        "task12_first": [1.0],
        "fused_first": [-1.0],
        "fused_second": [1.0],
    }
    results[0]["per_position_medians_ms"] = {
        "task12_first": 9.0,
        "fused_first": -1.0,
        "fused_second": 1.0,
    }

    failures = evaluate(results)

    assert any("ABBA evidence must contain exactly" in failure for failure in failures)
    assert any(
        "timing samples must be finite and non-negative" in failure
        for failure in failures
    )
    assert any(
        "position median does not match samples" in failure for failure in failures
    )


def test_gate_rejects_large_bucket_regression_and_mode_accuracy_mismatch() -> None:
    results = _complete_results()
    for result in results:
        if (
            result["world_size"] == 4
            and result["bucket_mib"] == 64
            and result["output_mode"] == "caller"
        ):
            result["fused_ms"] = float(result["task12_ms"]) * 1.01
        if (
            result["world_size"] == 2
            and result["bucket_mib"] == 16
            and result["output_mode"] == "lease"
        ):
            result["relative_l2"] = 0.006

    failures = evaluate(results)

    assert any(
        "4gpu/64MiB/caller: large-bucket latency regressed" in failure
        for failure in failures
    )
    assert any(
        "2gpu/16MiB: caller/lease accuracy mismatch" in failure for failure in failures
    )
