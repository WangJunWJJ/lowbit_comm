from __future__ import annotations

import pytest

from tests.benchmarks.assert_performance_gate import compare_results
from tests.benchmarks.result_schema import resolve_benchmark_identity, validate_result


def _result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "commit": "1f057cc",
        "hostname": "a6000",
        "gpu_name": "NVIDIA RTX A6000",
        "cuda_version": "12.1",
        "torch_version": "2.4",
        "world_size": 4,
        "dtype": "fp16",
        "numel": 8_388_608,
        "strategy": "all_gather",
        "latency_ms": 1.0,
        "effective_gbps": 10.0,
        "peak_memory_bytes": 1,
        "relative_l2": 0.01,
        "max_abs_error": 0.01,
        "rmse": 0.001,
        "non_finite": 0,
    }
    result.update(overrides)
    return result


def test_result_schema_accepts_reproducible_result() -> None:
    validate_result(_result())


def test_result_schema_reports_missing_fields() -> None:
    payload = _result()
    del payload["commit"]

    with pytest.raises(ValueError, match="commit"):
        validate_result(payload)


def test_result_schema_rejects_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        validate_result(_result(latency_ms=0.0))
    with pytest.raises(ValueError, match="non_finite"):
        validate_result(_result(non_finite=-1))


def test_result_schema_rejects_unknown_source_revision() -> None:
    with pytest.raises(ValueError, match="commit"):
        validate_result(_result(commit="unknown"))


def test_benchmark_identity_honors_container_overrides() -> None:
    identity = resolve_benchmark_identity(
        {
            "CCDL_BENCHMARK_COMMIT": "305e917",
            "CCDL_BENCHMARK_HOSTNAME": "user-SYS-6049GP-TRT-LongJing-Server",
        }
    )

    assert identity == {
        "commit": "305e917",
        "hostname": "user-SYS-6049GP-TRT-LongJing-Server",
    }


def test_performance_gate_reports_latency_regression() -> None:
    failures = compare_results(_result(latency_ms=1.0), _result(latency_ms=1.03), max_regression=0.02)

    assert failures == ["latency regression: 1.0300"]


def test_performance_gate_accepts_threshold_boundary() -> None:
    failures = compare_results(_result(latency_ms=1.0), _result(latency_ms=1.02), max_regression=0.02)

    assert failures == []


def test_performance_gate_rejects_incomparable_runs() -> None:
    failures = compare_results(_result(), _result(world_size=2), max_regression=0.02)

    assert failures == ["incomparable field world_size: baseline=4, candidate=2"]
