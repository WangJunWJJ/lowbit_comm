from __future__ import annotations

import pytest

from ccdl_comm import ExecutionInfo
from ccdl_comm.execution_info import ExecutionCounters, FallbackRecord


def _execution_info(**overrides: object) -> ExecutionInfo:
    values: dict[str, object] = {
        "requested_strategy": "ring",
        "executed_strategy": "ring",
        "backend": "cuda",
        "fallback_used": False,
        "fallback_reason": None,
        "stage_names": ("reduce",),
        "original_bytes": 2048,
        "compressed_bytes": 1024,
        "compression_ratio": 2.0,
        "workspace_cache_hit": True,
        "async_capable": True,
        "fast_path": "cuda_fused",
    }
    values.update(overrides)
    return ExecutionInfo(**values)


def test_execution_info_contains_fr015_fields_and_is_immutable() -> None:
    info = _execution_info()

    assert info.compression_ratio == 2.0
    with pytest.raises(Exception):
        info.fast_path = "fallback"  # type: ignore[misc]


def test_execution_info_copies_extension_details() -> None:
    details = {"kernel": "quant_pack"}
    info = _execution_info(details=details)
    details.clear()

    assert info.details == {"kernel": "quant_pack"}
    with pytest.raises(TypeError):
        info.details["kernel"] = "other"  # type: ignore[index]


def test_execution_info_rejects_inconsistent_fallback_and_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="fallback_reason"):
        _execution_info(fallback_used=True, fallback_reason=None)
    with pytest.raises(ValueError, match="original_bytes"):
        _execution_info(original_bytes=-1)
    with pytest.raises(ValueError, match="compression_ratio"):
        _execution_info(compression_ratio=0.0)


def test_execution_counters_expose_structured_runtime_fallback() -> None:
    counters = ExecutionCounters()
    record = FallbackRecord(
        reason="runtime layout is unsupported",
        from_path="cuda_fused",
        to_path="python_fallback",
    )

    counters._record_fallback(record)

    snapshot = counters.snapshot()
    assert snapshot.fallback_runs == 1
    assert snapshot.last_fallback is record
