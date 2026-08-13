from __future__ import annotations

import json

import pytest

from tests.benchmarks.sharded_parameter_pipeline_gate import GateFailure, evaluate


MODES = ("native_ddp", "full_fused", "sharded_fp", "sharded_compressed")


def write_trials(
    root,
    *,
    native=(100.0, 101.0, 99.0),
    compressed=(106.0, 107.0, 105.0),
    rank_diff: float = 0.0,
    compressed_fallback: str | None = None,
) -> None:
    throughputs = {
        "native_ddp": native,
        "full_fused": (102.0, 103.0, 101.0),
        "sharded_fp": (103.0, 104.0, 102.0),
        "sharded_compressed": compressed,
    }
    for mode in MODES:
        for trial, throughput in enumerate(throughputs[mode], start=1):
            payload = {
                "mode": mode,
                "world_size": 4,
                "timing": {"throughput_samples_per_second": throughput},
                "loss": {"final": 1.0},
                "correctness": {
                    "finite_loss": True,
                    "max_parameter_difference": (
                        rank_diff if mode == "sharded_compressed" else 0.0
                    ),
                },
                "fallback_reason": (
                    compressed_fallback if mode == "sharded_compressed" else None
                ),
                "selected_fast_path": (
                    "compressed_parameter_restore"
                    if mode == "sharded_compressed"
                    else mode
                ),
                "benchmark_environment": {
                    "physical_gpu_ids": [1, 2, 3, 4],
                    "image_id": "sha256:image",
                    "source_hash": "abc123",
                    "extension_sha256": "def456",
                },
            }
            path = root / f"{mode}-trial{trial}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")


def test_gate_uses_trial_median_and_requires_exact_rank_equality(tmp_path) -> None:
    write_trials(tmp_path)

    result = evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.05)

    assert result["native_median"] == 100.0
    assert result["compressed_median"] == 106.0
    assert result["speedup_vs_native"] == pytest.approx(1.06)
    assert result["passed"] is True


def test_gate_rejects_any_nonzero_rank_difference(tmp_path) -> None:
    write_trials(tmp_path, rank_diff=1e-7)

    with pytest.raises(GateFailure, match="rank parameter difference"):
        evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.0)


def test_gate_rejects_compressed_fallback(tmp_path) -> None:
    write_trials(tmp_path, compressed_fallback="native fallback")

    with pytest.raises(GateFailure, match="fallback"):
        evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.0)


def test_gate_requires_three_trials_per_mode(tmp_path) -> None:
    write_trials(tmp_path)
    (tmp_path / "full_fused-trial3.json").unlink()

    with pytest.raises(GateFailure, match="three trials"):
        evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.0)


def test_gate_rejects_environment_mismatch(tmp_path) -> None:
    write_trials(tmp_path)
    path = tmp_path / "native_ddp-trial1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["benchmark_environment"]["image_id"] = "sha256:other"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GateFailure, match="environment"):
        evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.0)


def test_gate_rejects_speedup_below_threshold(tmp_path) -> None:
    write_trials(tmp_path, compressed=(104.0, 104.0, 104.0))

    with pytest.raises(GateFailure, match="speedup"):
        evaluate(tmp_path, world_size=4, min_speedup_vs_native=1.05)
