"""Correctness-first gate for comparable native/sync/async training JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Sequence


_COMPRESSED_STRATEGIES = frozenset(
    {"all_gather", "all_reduce", "reduce_scatter", "hierarchical", "topology"}
)
_WORKLOAD_FIELDS = frozenset(
    {
        "synthetic",
        "data_root",
        "steps",
        "warmup_steps",
        "batch_size_per_rank",
        "input_dim",
        "hidden_dim",
        "depth",
        "num_classes",
        "learning_rate",
        "seed",
        "device",
        "dtype",
        "bit",
        "group_size",
        "error_feedback",
        "bucket_cap_mb",
    }
)


class GateFailure(AssertionError):
    """A benchmark result failed before a valid speedup claim could be made."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"{stage}: {message}")
        self.stage = stage


@dataclass(frozen=True, slots=True)
class GateThresholds:
    max_relative_loss_difference: float = 0.02
    max_gradient_relative_l2: float = 0.05
    min_async_sync_speedup: float = 1.0
    min_async_native_speedup: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_relative_loss_difference",
            "max_gradient_relative_l2",
            "min_async_sync_speedup",
            "min_async_native_speedup",
        ):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_relative_loss_difference < 0:
            raise ValueError("max_relative_loss_difference must be >= 0")
        if self.max_gradient_relative_l2 < 0:
            raise ValueError("max_gradient_relative_l2 must be >= 0")
        if self.min_async_sync_speedup < 1.0:
            raise ValueError("min_async_sync_speedup must be >= 1")
        if self.min_async_native_speedup < 1.0:
            raise ValueError("min_async_native_speedup must be >= 1")


def evaluate_runs(
    native: dict[str, Any],
    sync: dict[str, Any],
    async_run: dict[str, Any],
    *,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or GateThresholds()
    runs = {
        "native_ddp": native,
        "ccdl_sync": sync,
        "ccdl_async": async_run,
    }
    try:
        _validate_modes(runs)
        _validate_comparability(runs)
        _validate_correctness(runs, limits)
        _validate_loss(native, sync, async_run, limits)
        _validate_execution(sync, async_run)
        _validate_async_timeline(async_run)

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
            "schema",
            f"invalid or missing benchmark field: {type(exc).__name__}: {exc}",
        ) from exc

    async_vs_sync = throughputs["ccdl_async"] / throughputs["ccdl_sync"]
    if async_vs_sync <= limits.min_async_sync_speedup:
        raise GateFailure(
            "performance",
            "overlap benefit is absent: async throughput must be greater than "
            f"sync throughput (ratio={async_vs_sync:.6f}, "
            f"required>{limits.min_async_sync_speedup:.6f})",
        )
    async_vs_native = throughputs["ccdl_async"] / throughputs["native_ddp"]
    if async_vs_native <= limits.min_async_native_speedup:
        raise GateFailure(
            "performance",
            "end-to-end benefit is absent: async throughput must be greater than "
            f"native DDP throughput (ratio={async_vs_native:.6f}, "
            f"required>{limits.min_async_native_speedup:.6f})",
        )
    return {
        "passed": True,
        "thresholds": asdict(limits),
        "speedup": {
            "sync_vs_native": throughputs["ccdl_sync"] / throughputs["native_ddp"],
            "async_vs_sync": async_vs_sync,
            "async_vs_native": async_vs_native,
        },
        "runs": runs,
    }


def _validate_modes(runs: dict[str, dict[str, Any]]) -> None:
    for expected, run in runs.items():
        if run.get("mode") != expected:
            raise GateFailure(
                "schema",
                f"expected mode {expected!r}, received {run.get('mode')!r}",
            )


def _validate_comparability(runs: dict[str, dict[str, Any]]) -> None:
    fields = {
        "schema_version": lambda run: run["schema_version"],
        "world_size": lambda run: run["world_size"],
        "global_batch_size": lambda run: run["global_batch_size"],
        "parameter_count": lambda run: run["parameter_count"],
        "measured_steps": lambda run: run["timing"]["measured_steps"],
    }
    for name, getter in fields.items():
        values = {mode: getter(run) for mode, run in runs.items()}
        normalized = {
            mode: _positive_integer(value, name=name)
            for mode, value in values.items()
        }
        if len(set(normalized.values())) != 1:
            raise GateFailure(
                "comparability",
                f"{name} differs across runs: {normalized}",
            )
        if name == "schema_version" and next(iter(normalized.values())) != 2:
            raise GateFailure("comparability", "schema_version must be 2")
    workloads = {mode: run["workload"] for mode, run in runs.items()}
    if any(not isinstance(value, dict) or not value for value in workloads.values()):
        raise GateFailure("comparability", "workload must be a non-empty object")
    for workload in workloads.values():
        _validate_workload(workload)
    if any(workload != workloads["native_ddp"] for workload in workloads.values()):
        raise GateFailure(
            "comparability",
            f"workload differs across runs: {workloads}",
        )
    for mode, run in runs.items():
        workload = workloads[mode]
        expected_steps = workload["steps"] - workload["warmup_steps"]
        if run["timing"]["measured_steps"] != expected_steps:
            raise GateFailure(
                "comparability",
                f"{mode} measured_steps must equal steps - warmup_steps",
            )
        expected_batch = workload["batch_size_per_rank"] * run["world_size"]
        if run["global_batch_size"] != expected_batch:
            raise GateFailure(
                "comparability",
                f"{mode} global_batch_size must equal batch_size_per_rank * world_size",
            )


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateFailure("comparability", f"{name} must be a positive integer")
    return value


def _validate_workload(workload: dict[str, Any]) -> None:
    missing = sorted(_WORKLOAD_FIELDS.difference(workload))
    if missing:
        raise GateFailure(
            "comparability",
            f"workload missing required fields: {missing}",
        )
    synthetic = _strict_bool(workload["synthetic"], name="workload.synthetic")
    _strict_bool(workload["error_feedback"], name="workload.error_feedback")
    data_root = workload["data_root"]
    if data_root is not None and not isinstance(data_root, str):
        raise GateFailure("comparability", "workload.data_root must be string or null")
    if synthetic != (data_root is None) or data_root == "":
        raise GateFailure(
            "comparability",
            "workload dataset selection requires synthetic=true with data_root=null "
            "or synthetic=false with a non-empty data_root",
        )
    for name in (
        "steps",
        "batch_size_per_rank",
        "input_dim",
        "hidden_dim",
        "depth",
        "num_classes",
        "group_size",
        "bucket_cap_mb",
    ):
        _positive_integer(workload[name], name=f"workload.{name}")
    warmup_steps = workload["warmup_steps"]
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
        or warmup_steps >= workload["steps"]
    ):
        raise GateFailure(
            "comparability",
            "workload.warmup_steps must be an integer in [0, steps)",
        )
    seed = workload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GateFailure("comparability", "workload.seed must be an integer")
    if isinstance(workload["learning_rate"], bool) or not isinstance(
        workload["learning_rate"], (int, float)
    ):
        raise GateFailure(
            "comparability",
            "workload.learning_rate must be a finite positive number",
        )
    _finite_positive(
        workload["learning_rate"],
        stage="comparability",
        name="workload.learning_rate",
    )
    if workload["device"] not in {"auto", "cpu", "cuda"}:
        raise GateFailure("comparability", "workload.device is invalid")
    if workload["dtype"] not in {"fp16", "bf16", "fp32"}:
        raise GateFailure("comparability", "workload.dtype is invalid")
    if (
        isinstance(workload["bit"], bool)
        or not isinstance(workload["bit"], int)
        or workload["bit"] not in {4, 8}
    ):
        raise GateFailure("comparability", "workload.bit must be 4 or 8")


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise GateFailure("schema", f"{name} must be a boolean")
    return value


def _validate_correctness(
    runs: dict[str, dict[str, Any]],
    thresholds: GateThresholds,
) -> None:
    for mode, run in runs.items():
        correctness = run.get("correctness", {})
        finite_loss = correctness.get("finite_loss")
        if not isinstance(finite_loss, bool) or not finite_loss:
            raise GateFailure(
                "correctness",
                f"{mode} finite_loss must be boolean true",
            )
        rank_consistent = correctness.get("rank_parameters_consistent")
        if not isinstance(rank_consistent, bool) or not rank_consistent:
            raise GateFailure(
                "correctness",
                f"{mode} rank_parameters_consistent must be boolean true; "
                "rank parameters are inconsistent",
            )
        maximum_difference = _finite_nonnegative(
            correctness.get("max_parameter_difference"),
            stage="correctness",
            name=f"{mode} max_parameter_difference",
        )
        if maximum_difference != 0.0:
            raise GateFailure(
                "correctness",
                f"{mode} max_parameter_difference must be zero when ranks are consistent",
            )
        gradient_error = correctness.get("gradient_relative_l2")
        if gradient_error is not None:
            gradient_error = _finite_nonnegative(
                gradient_error,
                stage="correctness",
                name=f"{mode} gradient_relative_l2",
            )
            if gradient_error > thresholds.max_gradient_relative_l2:
                raise GateFailure(
                    "correctness",
                    f"{mode} gradient relative L2 {gradient_error:.6f} exceeds "
                    f"{thresholds.max_gradient_relative_l2:.6f}",
                )


def _validate_loss(
    native: dict[str, Any],
    sync: dict[str, Any],
    async_run: dict[str, Any],
    thresholds: GateThresholds,
) -> None:
    reference = _finite_number(
        native["loss"]["final"],
        stage="convergence",
        name="native final loss",
    )
    for mode, run in (
        ("native_ddp", native),
        ("ccdl_sync", sync),
        ("ccdl_async", async_run),
    ):
        initial_loss = _finite_number(
            run["loss"]["initial"],
            stage="convergence",
            name=f"{mode} initial loss",
        )
        candidate_loss = _finite_number(
            run["loss"]["final"],
            stage="convergence",
            name=f"{mode} final loss",
        )
        if candidate_loss > initial_loss:
            raise GateFailure(
                "convergence",
                f"{mode} loss did not decrease: initial={initial_loss:.6f}, "
                f"final={candidate_loss:.6f}",
            )
        if mode == "native_ddp":
            continue
        relative = abs(candidate_loss - reference) / max(abs(reference), 1e-12)
        if relative > thresholds.max_relative_loss_difference:
            raise GateFailure(
                "convergence",
                f"{mode} loss divergence {relative:.6f} exceeds "
                f"{thresholds.max_relative_loss_difference:.6f}",
            )


def _validate_execution(sync: dict[str, Any], async_run: dict[str, Any]) -> None:
    for mode, run in (("ccdl_sync", sync), ("ccdl_async", async_run)):
        execution = run["execution"]
        fallback = execution["fallback_reason"]
        if fallback is not None:
            raise GateFailure(
                "execution",
                f"{mode} fallback_reason must be null, received {fallback!r}",
            )
        strategy = str(execution["effective_strategy"]).strip()
        if strategy not in _COMPRESSED_STRATEGIES:
            raise GateFailure(
                "execution",
                f"{mode} did not use an allowed compressed strategy: {strategy!r}",
            )
        if execution.get("capability") != "cuda_extension":
            raise GateFailure(
                "execution",
                f"{mode} requires CUDA extension capability for this performance gate",
            )


def _validate_async_timeline(async_run: dict[str, Any]) -> None:
    timing = async_run["timing"]
    if timing.get("overlap_classification") != "timeline_overlapped":
        raise GateFailure(
            "async_semantics",
            "timeline evidence is required before evaluating async performance",
        )
    efficiency = _finite_nonnegative(
        timing.get("overlap_efficiency", 0.0),
        stage="async_semantics",
        name="overlap_efficiency",
    )
    if efficiency <= 0 or efficiency > 1:
        raise GateFailure(
            "async_semantics",
            f"timeline overlap efficiency must be in (0, 1], got {efficiency}",
        )
    communication = _finite_positive(
        timing["communication_ms"],
        stage="async_semantics",
        name="communication_ms",
    )
    compute = _finite_positive(
        timing["compute_ms"],
        stage="async_semantics",
        name="compute_ms",
    )
    union = _finite_nonnegative(
        timing["overlapped_ms"],
        stage="async_semantics",
        name="overlapped_ms",
    )
    exposed = _finite_nonnegative(
        timing["exposed_communication_ms"],
        stage="async_semantics",
        name="exposed_communication_ms",
    )
    tolerance = max(1e-6, 1e-4 * max(communication, compute, union, 1.0))
    if exposed > communication + tolerance:
        raise GateFailure(
            "async_semantics",
            "exposed communication cannot exceed total communication",
        )
    expected_union = compute + exposed
    if abs(union - expected_union) > tolerance:
        raise GateFailure(
            "async_semantics",
            f"timeline union is inconsistent: observed={union:.6f}, "
            f"expected={expected_union:.6f}",
        )
    intersection = max(0.0, communication - exposed)
    expected_efficiency = intersection / min(communication, compute)
    if abs(efficiency - expected_efficiency) > 1e-4:
        raise GateFailure(
            "async_semantics",
            f"overlap efficiency is inconsistent: observed={efficiency:.6f}, "
            f"expected={expected_efficiency:.6f}",
        )


def _finite_positive(value: Any, *, stage: str, name: str) -> float:
    result = _finite_number(value, stage=stage, name=name)
    if not isfinite(result) or result <= 0:
        raise GateFailure(stage, f"{name} must be finite and > 0")
    return result


def _finite_nonnegative(value: Any, *, stage: str, name: str) -> float:
    result = _finite_number(value, stage=stage, name=name)
    if not isfinite(result) or result < 0:
        raise GateFailure(stage, f"{name} must be finite and >= 0")
    return result


def _finite_number(value: Any, *, stage: str, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GateFailure(stage, f"{name} must be a finite number") from exc
    if not isfinite(result):
        raise GateFailure(stage, f"{name} must be a finite number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--sync", type=Path, required=True)
    parser.add_argument("--async", dest="async_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-loss-difference", type=float, default=0.02)
    parser.add_argument("--max-gradient-relative-l2", type=float, default=0.05)
    parser.add_argument("--min-async-sync-speedup", type=float, default=1.0)
    parser.add_argument("--min-async-native-speedup", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs: dict[str, dict[str, Any]] = {}
    limits: GateThresholds | None = None
    exit_code = 0
    try:
        runs = {
            "native_ddp": _read_json(args.native),
            "ccdl_sync": _read_json(args.sync),
            "ccdl_async": _read_json(args.async_path),
        }
        limits = GateThresholds(
            max_relative_loss_difference=args.max_relative_loss_difference,
            max_gradient_relative_l2=args.max_gradient_relative_l2,
            min_async_sync_speedup=args.min_async_sync_speedup,
            min_async_native_speedup=args.min_async_native_speedup,
        )
        report = evaluate_runs(
            runs["native_ddp"],
            runs["ccdl_sync"],
            runs["ccdl_async"],
            thresholds=limits,
        )
    except GateFailure as exc:
        exit_code = 1
        report = {
            "passed": False,
            "failure_stage": exc.stage,
            "failure": str(exc),
            "thresholds": asdict(limits) if limits is not None else None,
            "runs": runs,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        exit_code = 1
        report = {
            "passed": False,
            "failure_stage": "input",
            "failure": f"{type(exc).__name__}: {exc}",
            "thresholds": asdict(limits) if limits is not None else None,
            "runs": runs,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return exit_code


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"benchmark JSON must contain an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
