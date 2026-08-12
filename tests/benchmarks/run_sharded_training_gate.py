"""Correctness-first gate for ReducedShard end-to-end training evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.training.sharded_metrics import PHASE_NAMES


_MODES = ("native_ddp", "ccdl_full_gradient", "ccdl_sharded_sgd")
_POINTER_NAMES = frozenset(
    {
        "flat_gradients",
        "reduced_gradient",
        "local_parameters",
        "gathered_parameters",
    }
)


class GateFailure(AssertionError):
    """Evidence failed at a named, persistable validation stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class ShardedGateThresholds:
    max_relative_loss_difference: float = 0.02
    min_2gpu_native_ratio: float = 0.95
    min_4gpu_full_gradient_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_relative_loss_difference",
            "min_2gpu_native_ratio",
            "min_4gpu_full_gradient_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.max_relative_loss_difference <= 0.02:
            raise ValueError("max_relative_loss_difference must be in [0, 0.02]")
        if self.min_2gpu_native_ratio < 0.95:
            raise ValueError("min_2gpu_native_ratio must be >= 0.95")
        if self.min_4gpu_full_gradient_ratio < 1.0:
            raise ValueError("min_4gpu_full_gradient_ratio must be >= 1.0")


def evaluate_sharded_runs(
    native: dict[str, Any],
    full_gradient: dict[str, Any],
    sharded: dict[str, Any],
    *,
    thresholds: ShardedGateThresholds | None = None,
) -> dict[str, Any]:
    """Validate comparable evidence before calculating a performance claim."""

    limits = thresholds or ShardedGateThresholds()
    runs = dict(zip(_MODES, (native, full_gradient, sharded), strict=True))
    try:
        _validate_comparability(runs)
        _validate_correctness(runs)
        _validate_execution(runs)
        _validate_convergence(runs, limits)
        throughputs = {
            mode: _finite_positive(
                run["timing"]["throughput_samples_per_second"],
                stage="performance",
                name=f"{mode} throughput",
            )
            for mode, run in runs.items()
        }
    except GateFailure:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise GateFailure(
            "input",
            f"invalid or missing benchmark field: {type(exc).__name__}: {exc}",
        ) from exc

    world_size = native["world_size"]
    sharded_vs_native = throughputs["ccdl_sharded_sgd"] / throughputs["native_ddp"]
    sharded_vs_full = (
        throughputs["ccdl_sharded_sgd"] / throughputs["ccdl_full_gradient"]
    )
    if world_size == 2 and sharded_vs_native < limits.min_2gpu_native_ratio:
        raise GateFailure(
            "performance",
            "2-GPU native DDP ratio is below the floor: "
            f"{sharded_vs_native:.6f} < {limits.min_2gpu_native_ratio:.6f}",
        )
    if world_size == 4 and sharded_vs_full <= limits.min_4gpu_full_gradient_ratio:
        raise GateFailure(
            "performance",
            "4-GPU sharded consumer benefit is absent: "
            f"{sharded_vs_full:.6f} <= {limits.min_4gpu_full_gradient_ratio:.6f}",
        )
    return {
        "passed": True,
        "thresholds": asdict(limits),
        "speedup": {
            "full_gradient_vs_native": (
                throughputs["ccdl_full_gradient"] / throughputs["native_ddp"]
            ),
            "sharded_vs_native": sharded_vs_native,
            "sharded_vs_full_gradient": sharded_vs_full,
        },
        "runs": runs,
    }


def _validate_comparability(runs: dict[str, dict[str, Any]]) -> None:
    for mode, run in runs.items():
        if not isinstance(run, dict) or run.get("mode") != mode:
            raise GateFailure("comparability", f"expected mode {mode!r}")
    for name, getter in {
        "schema_version": lambda run: run["schema_version"],
        "world_size": lambda run: run["world_size"],
        "global_batch_size": lambda run: run["global_batch_size"],
        "parameter_count": lambda run: run["parameter_count"],
        "measured_steps": lambda run: run["timing"]["measured_steps"],
    }.items():
        values = {mode: _positive_integer(getter(run), name) for mode, run in runs.items()}
        if len(set(values.values())) != 1:
            raise GateFailure("comparability", f"{name} differs across runs: {values}")
    if next(iter(runs.values()))["schema_version"] != 3:
        raise GateFailure("comparability", "schema_version must be 3")
    world_size = next(iter(runs.values()))["world_size"]
    if world_size not in {2, 4}:
        raise GateFailure("comparability", "world_size must be 2 or 4")
    workloads = {mode: run["workload"] for mode, run in runs.items()}
    if any(not isinstance(value, dict) or not value for value in workloads.values()):
        raise GateFailure("comparability", "workload must be a non-empty object")
    if any(value != workloads["native_ddp"] for value in workloads.values()):
        raise GateFailure("comparability", "workload differs across runs")
    workload = workloads["native_ddp"]
    measured_steps = workload["steps"] - workload["warmup_steps"]
    expected_batch = workload["batch_size_per_rank"] * world_size
    for mode, run in runs.items():
        if run["timing"]["measured_steps"] != measured_steps:
            raise GateFailure(
                "comparability", f"{mode} measured_steps does not match workload"
            )
        if run["global_batch_size"] != expected_batch:
            raise GateFailure(
                "comparability", f"{mode} global batch does not match workload"
            )


def _validate_correctness(runs: dict[str, dict[str, Any]]) -> None:
    for mode, run in runs.items():
        correctness = run["correctness"]
        if _strict_bool(
            correctness["finite_loss"],
            f"{mode}.finite_loss",
            stage="correctness",
        ) is not True:
            raise GateFailure("correctness", f"{mode} reported non-finite loss")
        if _strict_bool(
            correctness["rank_parameters_consistent"],
            f"{mode}.rank_parameters_consistent",
            stage="correctness",
        ) is not True:
            raise GateFailure("correctness", f"{mode} rank parameters differ")
        difference = _finite_number(
            correctness["max_parameter_difference"],
            stage="correctness",
            name=f"{mode} max parameter difference",
        )
        if difference != 0.0:
            raise GateFailure("correctness", f"{mode} max parameter difference is nonzero")


def _validate_execution(runs: dict[str, dict[str, Any]]) -> None:
    for mode in ("ccdl_full_gradient", "ccdl_sharded_sgd"):
        execution = runs[mode]["execution"]
        if execution["fallback_reason"] is not None:
            raise GateFailure("execution", f"{mode} used fallback")
        if execution["capability"] != "cuda_extension":
            raise GateFailure("execution", f"{mode} did not use the CUDA extension")
    sharded = runs["ccdl_sharded_sgd"]
    execution = sharded["execution"]
    if execution["effective_strategy"] != "compressed":
        raise GateFailure("execution", "sharded mode did not execute compressed transport")
    if execution.get("fast_path") != "cuda_reduced_shard":
        raise GateFailure("execution", "sharded mode did not use cuda_reduced_shard")
    if execution.get("output_layout") != "shard":
        raise GateFailure("execution", "sharded mode did not return shard layout")
    if _strict_bool(
        sharded["phase_timing_measured"],
        "phase_timing_measured",
        stage="execution",
    ) is not True:
        raise GateFailure("execution", "sharded phase timing was not measured")
    phases = sharded["phase_timing_ms"]
    if set(phases) != set(PHASE_NAMES):
        raise GateFailure("execution", "sharded phase timing is incomplete")
    for name in PHASE_NAMES:
        value = _finite_number(phases[name], stage="execution", name=name)
        if value < 0:
            raise GateFailure("execution", f"{name} must be >= 0")
    reuse = sharded["buffer_reuse"]
    if _strict_bool(
        reuse["stable"],
        "buffer_reuse.stable",
        stage="execution",
    ) is not True:
        raise GateFailure("execution", "caller-owned buffers were not stable")
    initial = reuse["initial_pointers"]
    final = reuse["final_pointers"]
    if set(initial) != _POINTER_NAMES or initial != final:
        raise GateFailure("execution", "caller-owned buffer pointers changed")
    if any(isinstance(pointer, bool) or not isinstance(pointer, int) or pointer <= 0 for pointer in initial.values()):
        raise GateFailure("execution", "buffer pointers must be positive integers")


def _validate_convergence(
    runs: dict[str, dict[str, Any]],
    thresholds: ShardedGateThresholds,
) -> None:
    final_losses = {}
    for mode, run in runs.items():
        loss = run["loss"]
        initial = _finite_number(loss["initial"], stage="convergence", name=f"{mode} initial loss")
        final = _finite_number(loss["final"], stage="convergence", name=f"{mode} final loss")
        samples = loss["samples"]
        if not isinstance(samples, list) or not samples:
            raise GateFailure("convergence", f"{mode} loss samples must be non-empty")
        for index, sample in enumerate(samples):
            _finite_number(sample, stage="convergence", name=f"{mode} loss sample {index}")
        if final > initial:
            raise GateFailure("convergence", f"{mode} loss did not decrease")
        final_losses[mode] = final
    native_final = final_losses["native_ddp"]
    denominator = max(abs(native_final), 1e-12)
    for mode in ("ccdl_full_gradient", "ccdl_sharded_sgd"):
        relative = abs(final_losses[mode] - native_final) / denominator
        if relative > thresholds.max_relative_loss_difference:
            raise GateFailure(
                "convergence",
                f"{mode} relative final loss difference {relative:.6f} exceeds "
                f"{thresholds.max_relative_loss_difference:.6f}",
            )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateFailure("comparability", f"{name} must be a positive integer")
    return value


def _strict_bool(value: Any, name: str, *, stage: str) -> bool:
    if not isinstance(value, bool):
        raise GateFailure(stage, f"{name} must be a boolean")
    return value


def _finite_number(value: Any, *, stage: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise GateFailure(stage, f"{name} must be finite")
    return float(value)


def _finite_positive(value: Any, *, stage: str, name: str) -> float:
    result = _finite_number(value, stage=stage, name=name)
    if result <= 0:
        raise GateFailure(stage, f"{name} must be > 0")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--full-gradient", type=Path, required=True)
    parser.add_argument("--sharded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        "native": args.native,
        "full_gradient": args.full_gradient,
        "sharded": args.sharded,
    }
    raw_inputs: dict[str, Any] = {}
    try:
        for name, path in paths.items():
            raw = path.read_text(encoding="utf-8")
            raw_inputs[name] = raw
            raw_inputs[name] = json.loads(raw)
        report = evaluate_sharded_runs(
            raw_inputs["native"],
            raw_inputs["full_gradient"],
            raw_inputs["sharded"],
        )
        code = 0
    except GateFailure as exc:
        report = {
            "passed": False,
            "failure_stage": exc.stage,
            "failure": str(exc),
            "raw_inputs": raw_inputs,
        }
        code = 1
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        report = {
            "passed": False,
            "failure_stage": "input",
            "failure": f"{type(exc).__name__}: {exc}",
            "raw_inputs": raw_inputs,
        }
        code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
