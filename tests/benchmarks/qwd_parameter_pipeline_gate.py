"""Median-throughput and precision gate for qWD parameter communication."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Sequence


MODES = ("native_ddp", "sharded_fp", "sharded_compressed", "sharded_qwd")


class GateFailure(RuntimeError):
    """Raised when benchmark evidence cannot authorize qWD."""


def evaluate(
    results_dir: Path | str,
    *,
    world_size: int,
    min_speedup_vs_native: float = 1.05,
    min_ratio_vs_direct_int8: float = 0.98,
    max_loss_ratio_vs_sharded_fp: float = 1.02,
) -> dict[str, Any]:
    """Validate comparable trials and return qWD median ratios."""

    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    for name, value in (
        ("min_speedup_vs_native", min_speedup_vs_native),
        ("min_ratio_vs_direct_int8", min_ratio_vs_direct_int8),
        ("max_loss_ratio_vs_sharded_fp", max_loss_ratio_vs_sharded_fp),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")

    grouped: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for path in sorted(Path(results_dir).rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("mode")
        if mode in grouped and payload.get("world_size") == world_size:
            grouped[mode].append(payload)
    for mode, trials in grouped.items():
        if len(trials) < 3:
            raise GateFailure(f"{mode} must provide at least three trials")

    expected_environment: tuple[Any, ...] | None = None
    throughputs: dict[str, list[float]] = {mode: [] for mode in MODES}
    final_losses: dict[str, list[float]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for payload in grouped[mode]:
            environment = payload.get("benchmark_environment")
            if not isinstance(environment, dict):
                raise GateFailure("every trial must provide benchmark environment")
            signature = (
                tuple(environment.get("physical_gpu_ids", ())),
                environment.get("image_id"),
                environment.get("source_hash"),
                environment.get("extension_sha256"),
            )
            if expected_environment is None:
                expected_environment = signature
            elif signature != expected_environment:
                raise GateFailure("benchmark environment differs across trials")
            _validate_common_trial(mode, payload)
            throughputs[mode].append(
                _finite_positive(
                    payload.get("timing", {}).get(
                        "throughput_samples_per_second"
                    ),
                    f"{mode} throughput",
                )
            )
            final_losses[mode].append(
                _finite_nonnegative(
                    payload.get("loss", {}).get("final"),
                    f"{mode} final loss",
                )
            )
            if mode == "sharded_qwd":
                _validate_qwd_trial(payload)

    throughput_medians = {
        mode: median(values) for mode, values in throughputs.items()
    }
    loss_medians = {
        mode: median(values) for mode, values in final_losses.items()
    }
    qwd_speedup = (
        throughput_medians["sharded_qwd"] / throughput_medians["native_ddp"]
    )
    qwd_direct_ratio = (
        throughput_medians["sharded_qwd"]
        / throughput_medians["sharded_compressed"]
    )
    qwd_loss_ratio = loss_medians["sharded_qwd"] / max(
        loss_medians["sharded_fp"],
        1.0e-12,
    )
    if qwd_speedup < min_speedup_vs_native:
        raise GateFailure(
            f"qWD speedup {qwd_speedup:.6f} is below required "
            f"{min_speedup_vs_native:.6f}"
        )
    if qwd_direct_ratio < min_ratio_vs_direct_int8:
        raise GateFailure(
            f"qWD ratio versus direct INT8 {qwd_direct_ratio:.6f} is below "
            f"required {min_ratio_vs_direct_int8:.6f}"
        )
    if qwd_loss_ratio > max_loss_ratio_vs_sharded_fp:
        raise GateFailure(
            f"qWD loss ratio {qwd_loss_ratio:.6f} exceeds "
            f"{max_loss_ratio_vs_sharded_fp:.6f}"
        )
    assert expected_environment is not None
    return {
        "passed": True,
        "world_size": world_size,
        "trial_count": {mode: len(grouped[mode]) for mode in MODES},
        "throughput_medians": throughput_medians,
        "loss_medians": loss_medians,
        "qwd_speedup_vs_native": qwd_speedup,
        "qwd_ratio_vs_direct_int8": qwd_direct_ratio,
        "qwd_loss_ratio_vs_sharded_fp": qwd_loss_ratio,
        "benchmark_environment": {
            "physical_gpu_ids": list(expected_environment[0]),
            "image_id": expected_environment[1],
            "source_hash": expected_environment[2],
            "extension_sha256": expected_environment[3],
        },
    }


def _validate_common_trial(mode: str, payload: dict[str, Any]) -> None:
    correctness = payload.get("correctness")
    if not isinstance(correctness, dict):
        raise GateFailure(f"{mode} correctness evidence is missing")
    if correctness.get("finite_loss") is not True:
        raise GateFailure(f"{mode} produced a non-finite loss")
    difference = _finite_nonnegative(
        correctness.get("max_parameter_difference"),
        f"{mode} rank parameter difference",
    )
    if difference != 0.0:
        raise GateFailure(f"{mode} rank parameter difference must be exactly zero")


def _validate_qwd_trial(payload: dict[str, Any]) -> None:
    if payload.get("fallback_reason") is not None:
        raise GateFailure("sharded_qwd trial used fallback")
    if payload.get("selected_fast_path") != "fused_int8_qwd":
        raise GateFailure("sharded_qwd trial did not select fused qWD")
    workspace = payload.get("workspace_pointers")
    if not isinstance(workspace, dict) or workspace.get("stable") is not True:
        raise GateFailure("sharded_qwd workspace is not stable")
    communication = payload.get("parameter_communication")
    if not isinstance(communication, dict):
        raise GateFailure("sharded_qwd parameter communication evidence is missing")
    if communication.get("algorithm") != "qwd" or communication.get("bit") != 8:
        raise GateFailure("sharded_qwd must report INT8 qWD")
    decisions = communication.get("decision_counts")
    if not isinstance(decisions, dict) or int(decisions.get("qwd", 0)) < 1:
        raise GateFailure("sharded_qwd must report measured qWD decisions")
    errors = communication.get("sampled_relative_errors")
    if not isinstance(errors, list):
        raise GateFailure("sharded_qwd sampled relative errors must be a list")
    for value in errors:
        _finite_nonnegative(value, "sharded_qwd sampled relative error")


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise GateFailure(f"{name} must be > 0")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0:
        raise GateFailure(f"{name} must be >= 0")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateFailure(f"{name} must be finite")
    result = float(value)
    if not isfinite(result):
        raise GateFailure(f"{name} must be finite")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(args.results_dir, world_size=args.world_size)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
