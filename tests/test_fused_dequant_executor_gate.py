from tests.benchmarks.fused_dequant_executor_gate import evaluate


TASK0 = {
    (2, 1): 0.38,
    (2, 16): 1.81,
    (2, 64): 6.99,
    (4, 1): 0.45,
    (4, 16): 4.44,
    (4, 64): 17.24,
}


def _result(world_size: int, bucket_mib: int, **overrides):
    task0_ms = TASK0[(world_size, bucket_mib)]
    result = {
        "world_size": world_size,
        "bucket_mib": bucket_mib,
        "baseline_ms": task0_ms,
        "fused_ms": task0_ms * 0.98,
        "baseline_relative_l2": 0.0059,
        "fused_relative_l2": 0.0059,
        "steady_allocation_bytes": 0,
        "fast_path": "cuda_fused_dequant_reduce_mean_ef",
        "fallback_used": False,
        "measurement_order": "baseline-fused-fused-baseline",
        "error_feedback_reference": "local_reconstruction",
    }
    result.update(overrides)
    return result


def _complete_results():
    return [
        _result(world_size, bucket_mib)
        for world_size in (2, 4)
        for bucket_mib in (1, 16, 64)
        for _ in range(5)
    ]


def test_fused_dequant_gate_accepts_complete_positive_matrix() -> None:
    assert evaluate(_complete_results(), task0_baselines=TASK0) == []


def test_fused_dequant_gate_rejects_missing_case_and_insufficient_runs() -> None:
    results = _complete_results()
    results = [
        result
        for result in results
        if not (result["world_size"] == 4 and result["bucket_mib"] == 64)
    ]
    results.pop()

    failures = evaluate(results, task0_baselines=TASK0)

    assert any("missing 4gpu/64MiB" in failure for failure in failures)
    assert any("requires 5 runs" in failure for failure in failures)


def test_fused_dequant_gate_rejects_fallback_allocation_precision_and_fast_path() -> None:
    results = _complete_results()
    results[0].update(
        fallback_used=True,
        steady_allocation_bytes=1024,
        fused_relative_l2=0.0061,
        fast_path="python_fallback",
        measurement_order="baseline-fused",
    )

    failures = evaluate(results, task0_baselines=TASK0)

    assert any("fallback" in failure for failure in failures)
    assert any("steady allocation" in failure for failure in failures)
    assert any("precision mismatch" in failure for failure in failures)
    assert any("unexpected fast path" in failure for failure in failures)
    assert any("unbalanced measurement order" in failure for failure in failures)


def test_fused_dequant_gate_rejects_large_bucket_same_run_and_task0_regressions() -> None:
    same_run = _complete_results()
    for result in same_run:
        if result["world_size"] == 2 and result["bucket_mib"] == 16:
            result["fused_ms"] = result["baseline_ms"] * 1.01
    frozen = _complete_results()
    for result in frozen:
        if result["world_size"] == 4 and result["bucket_mib"] == 64:
            result["baseline_ms"] = TASK0[(4, 64)] * 1.20
            result["fused_ms"] = TASK0[(4, 64)] * 1.03

    same_run_failures = evaluate(same_run, task0_baselines=TASK0)
    frozen_failures = evaluate(frozen, task0_baselines=TASK0)

    assert any("latency regression" in failure for failure in same_run_failures)
    assert any("Task 0 regression" in failure for failure in frozen_failures)
