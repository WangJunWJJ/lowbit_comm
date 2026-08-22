"""Paired evidence contracts for the v0.4.0 PSI training matrix.

The module deliberately keeps schema, parity, timing, and transaction rules
CPU-only.  The distributed worker imports optional PSI/PyTorch dependencies and
uses these contracts as its fail-closed boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Any


ROUTES = ("native", "cag", "rsag_qwd")
PAIRED_SEEDS = (20260821, 20260822, 20260823)
SCHEMA_VERSION = 1
DEFAULT_PSI_SOURCE = "/home/user/wangjun/psi_policy_v040_three_route_20260822"

_STEP_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "attempt_id",
        "route",
        "seed",
        "epoch",
        "step",
        "batch_indices",
        "timing",
        "communication",
        "quality",
    }
)
_TIMING_FIELDS = frozenset(
    {
        "forward_s",
        "backward_s",
        "update_s",
        "communication_s",
        "validation_s",
        "report_serialization_s",
        "measured_s",
    }
)
_COMMUNICATION_FIELDS = frozenset(
    {
        "gradient_route",
        "parameter_route",
        "bytes",
        "qwd_s",
        "refresh_s",
        "decision",
    }
)
_QUALITY_FIELDS = frozenset(
    {
        "loss",
        "amp_scale",
        "learning_rate",
        "model_sha256",
        "rank_parameter_gap",
        "optimizer_step",
        "finite",
    }
)
_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "attempt_id",
        "route",
        "seed",
        "world_size",
        "physical_gpu_ids",
        "source_manifest_sha256",
        "data_sha256",
        "initial_parameter_sha256",
        "sampler_indices_sha256",
        "augmentation_rng_sha256",
        "lr_schedule_sha256",
        "amp_configuration",
        "batch_size_per_rank",
        "global_batch_size",
        "model_parameter_count",
        "epochs",
        "steps",
        "warmup_steps",
        "steady_samples_per_second",
        "step_latency_p50_ms",
        "step_latency_p95_ms",
        "epoch_time_s",
        "communication_time_s",
        "qwd_time_s",
        "refresh_time_s",
        "communication_bytes",
        "peak_memory_mib",
        "gpu_telemetry",
        "loss_trajectory",
        "validation_loss",
        "rank_gaps",
        "decision_counts",
        "failure_facts",
    }
)
_AMP_FIELDS = frozenset({"precision", "enabled", "initial_scale"})
_GPU_TELEMETRY_FIELDS = frozenset(
    {
        "gpu",
        "utilization",
        "memory_used_mib",
        "temperature_c",
        "sm_clock_mhz",
    }
)
_FAILURE_FACT_FIELDS = frozenset(
    {"phase", "category", "message", "rank", "step", "recoverable"}
)


@dataclass(frozen=True, slots=True)
class PairedRouteFacts:
    """Facts that must be identical among one seed's three routes."""

    initial_parameter_sha256: str
    sampler_indices: tuple[int, ...]
    augmentation_rng_sha256: str
    lr_schedule: tuple[float, ...]
    amp_configuration: tuple[str, bool, float]
    batch_size: int
    model_parameter_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.initial_parameter_sha256, "initial_parameter_sha256")
        _require_exact_int_tuple(self.sampler_indices, "sampler_indices")
        _require_sha256(self.augmentation_rng_sha256, "augmentation_rng_sha256")
        if type(self.lr_schedule) is not tuple or not all(
            _is_nonnegative_finite_float(value) for value in self.lr_schedule
        ):
            raise ValueError("lr_schedule must contain exact finite floats")
        if (
            type(self.amp_configuration) is not tuple
            or len(self.amp_configuration) != 3
            or type(self.amp_configuration[0]) is not str
            or self.amp_configuration[0] != "fp16"
            or type(self.amp_configuration[1]) is not bool
            or not _is_nonnegative_finite_float(self.amp_configuration[2])
        ):
            raise ValueError("amp_configuration is invalid")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(
            self.model_parameter_count,
            "model_parameter_count",
        )


@dataclass(frozen=True, slots=True)
class ResumeFacts:
    """Exact next-step facts used to prove checkpoint continuity."""

    next_batch_indices: tuple[int, ...]
    learning_rate: float
    amp_scale: float
    optimizer_state_sha256: str
    model_sha256: str
    next_loss: float
    post_learning_rate: float
    post_amp_scale: float
    post_optimizer_state_sha256: str
    post_model_sha256: str

    def __post_init__(self) -> None:
        _require_exact_int_tuple(self.next_batch_indices, "next_batch_indices")
        _require_nonnegative_float(self.learning_rate, "learning_rate")
        _require_nonnegative_float(self.amp_scale, "amp_scale")
        _require_sha256(self.optimizer_state_sha256, "optimizer_state_sha256")
        _require_sha256(self.model_sha256, "model_sha256")
        _require_finite_float(self.next_loss, "next_loss")
        _require_nonnegative_float(self.post_learning_rate, "post_learning_rate")
        _require_nonnegative_float(self.post_amp_scale, "post_amp_scale")
        _require_sha256(
            self.post_optimizer_state_sha256,
            "post_optimizer_state_sha256",
        )
        _require_sha256(self.post_model_sha256, "post_model_sha256")


@dataclass(frozen=True, slots=True)
class StepTiming:
    """Separated training and excluded timing domains for one step."""

    forward_s: float
    backward_s: float
    update_s: float
    communication_s: float
    validation_s: float = 0.0
    report_serialization_s: float = 0.0

    def __post_init__(self) -> None:
        for field in self.__slots__:
            _require_nonnegative_float(getattr(self, field), field)

    @property
    def measured_s(self) -> float:
        """Return only forward/backward/update/communication time."""
        return self.forward_s + self.backward_s + self.update_s + self.communication_s

    @property
    def total_wall_s(self) -> float:
        """Return training plus deliberately excluded work."""
        return self.measured_s + self.validation_s + self.report_serialization_s

    def to_dict(self) -> dict[str, float]:
        """Return the exact schema-v1 timing object."""
        return {
            "forward_s": self.forward_s,
            "backward_s": self.backward_s,
            "update_s": self.update_s,
            "communication_s": self.communication_s,
            "validation_s": self.validation_s,
            "report_serialization_s": self.report_serialization_s,
            "measured_s": self.measured_s,
        }


class RSAGQWDTransaction:
    """Publish optimizer/model candidates only after qWD Work succeeds."""

    __slots__ = ("_publish_model", "_publish_optimizer")

    def __init__(
        self,
        *,
        publish_optimizer: Callable[[object], None],
        publish_model: Callable[[object], None],
    ) -> None:
        if not callable(publish_optimizer) or not callable(publish_model):
            raise ValueError("transaction publishers must be callable")
        self._publish_optimizer = publish_optimizer
        self._publish_model = publish_model

    def commit(self, optimizer_candidate: object, qwd_work: object) -> object:
        """Wait first, then publish one indivisible ordered candidate pair."""
        wait = getattr(qwd_work, "wait", None)
        if not callable(wait):
            raise ValueError("qwd_work must provide callable wait()")
        model_candidate = wait()
        self._publish_optimizer(optimizer_candidate)
        self._publish_model(model_candidate)
        return model_candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the exact Task 5 worker contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument(
        "--seed", choices=PAIRED_SEEDS, type=int, default=PAIRED_SEEDS[0]
    )
    parser.add_argument("--psi-source", default=DEFAULT_PSI_SOURCE)
    parser.add_argument("--result-json", default="psi-v040-result.json")
    parser.add_argument("--raw-jsonl", default="psi-v040-steps.jsonl")
    parser.add_argument("--attempt-id", default="attempt-1")
    parser.add_argument("--checkpoint-dir", default="psi-v040-checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume-oracle-mode",
        choices=("off", "write", "require"),
        default="off",
    )
    parser.add_argument("--data-sha256", default="0" * 64)
    parser.add_argument("--psi-override", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp-initial-scale", type=float, default=1024.0)
    parser.add_argument("--smoke-midpoint", type=int, default=25)
    parser.add_argument("--inject-overflow-step", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args(argv)


def build_engine(
    route: object,
    *,
    native_factory: Callable[[], object],
    cag_factory: Callable[[], object],
    rsag_qwd_factory: Callable[[], object],
) -> object:
    """Instantiate exactly one route-specific update engine."""
    factories = {
        "native": native_factory,
        "cag": cag_factory,
        "rsag_qwd": rsag_qwd_factory,
    }
    if type(route) is not str or route not in factories:
        raise ValueError("route must be one exact approved route")
    factory = factories[route]
    if not callable(factory):
        raise ValueError("route factory must be callable")
    return factory()


def assert_paired_route_parity(
    facts_by_route: object,
) -> None:
    """Reject any paired input drift before a route starts training."""
    if type(facts_by_route) is not dict or set(facts_by_route) != set(ROUTES):
        raise ValueError("paired route fields must be exactly the approved routes")
    baseline = facts_by_route["native"]
    if type(baseline) is not PairedRouteFacts:
        raise ValueError("native parity facts are invalid")
    for route in ROUTES[1:]:
        candidate = facts_by_route[route]
        if type(candidate) is not PairedRouteFacts:
            raise ValueError(f"{route} parity facts are invalid")
        for field in PairedRouteFacts.__slots__:
            if getattr(candidate, field) != getattr(baseline, field):
                raise ValueError(f"paired route drift: {field}")


def assert_resume_matches(oracle: object, resumed: object) -> None:
    """Compare every required next-step resume fact exactly."""
    if type(oracle) is not ResumeFacts or type(resumed) is not ResumeFacts:
        raise ValueError("resume facts must be exact ResumeFacts")
    for field in ResumeFacts.__slots__:
        if getattr(oracle, field) != getattr(resumed, field):
            raise ValueError(f"resume drift: {field}")


def canonical_sha256(value: object) -> str:
    """Hash one JSON-safe fact with stable separators and key ordering."""
    encoded = dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_step_record(
    *,
    task_id: str,
    attempt_id: str,
    route: str,
    seed: int,
    epoch: int,
    step: int,
    batch_indices: tuple[int, ...],
    timing: StepTiming,
    gradient_route: str,
    parameter_route: str,
    communication_bytes: int,
    qwd_s: float,
    refresh_s: float,
    decision: str,
    loss: float,
    amp_scale: float,
    learning_rate: float,
    model_sha256: str,
    rank_parameter_gap: float,
    optimizer_step: int,
    finite: bool,
) -> dict[str, object]:
    """Build and freshly validate one raw per-step schema-v1 row."""
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "route": route,
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "batch_indices": list(batch_indices),
        "timing": timing.to_dict(),
        "communication": {
            "gradient_route": gradient_route,
            "parameter_route": parameter_route,
            "bytes": communication_bytes,
            "qwd_s": qwd_s,
            "refresh_s": refresh_s,
            "decision": decision,
        },
        "quality": {
            "loss": loss,
            "amp_scale": amp_scale,
            "learning_rate": learning_rate,
            "model_sha256": model_sha256,
            "rank_parameter_gap": rank_parameter_gap,
            "optimizer_step": optimizer_step,
            "finite": finite,
        },
    }
    return validate_step_record(record)


def validate_step_record(value: object) -> dict[str, object]:
    """Freshly validate one exact raw schema-v1 row."""
    record = _require_exact_dict(value, _STEP_FIELDS, "step")
    _require_schema_identity(record)
    _require_nonempty_str(record["task_id"], "task_id")
    _require_nonempty_str(record["attempt_id"], "attempt_id")
    _require_route(record["route"])
    _require_seed(record["seed"])
    _require_nonnegative_int(record["epoch"], "epoch")
    _require_nonnegative_int(record["step"], "step")
    _require_exact_int_list(record["batch_indices"], "batch_indices")

    timing = _require_exact_dict(record["timing"], _TIMING_FIELDS, "timing")
    for field in _TIMING_FIELDS:
        _require_nonnegative_float(timing[field], field)
    measured = sum(
        timing[field]
        for field in (
            "forward_s",
            "backward_s",
            "update_s",
            "communication_s",
        )
    )
    if timing["measured_s"] != measured:
        raise ValueError("timing measured_s is inconsistent")

    communication = _require_exact_dict(
        record["communication"],
        _COMMUNICATION_FIELDS,
        "communication",
    )
    _require_nonempty_str(communication["gradient_route"], "gradient_route")
    _require_nonempty_str(communication["parameter_route"], "parameter_route")
    _require_nonnegative_int(communication["bytes"], "bytes")
    _require_nonnegative_float(communication["qwd_s"], "qwd_s")
    _require_nonnegative_float(communication["refresh_s"], "refresh_s")
    _require_nonempty_str(communication["decision"], "decision")

    quality = _require_exact_dict(record["quality"], _QUALITY_FIELDS, "quality")
    _require_finite_float(quality["loss"], "loss")
    _require_nonnegative_float(quality["amp_scale"], "amp_scale")
    _require_nonnegative_float(quality["learning_rate"], "learning_rate")
    _require_sha256(quality["model_sha256"], "model_sha256")
    _require_nonnegative_float(quality["rank_parameter_gap"], "rank_parameter_gap")
    _require_nonnegative_int(quality["optimizer_step"], "optimizer_step")
    if type(quality["finite"]) is not bool:
        raise ValueError("finite must be an exact bool")
    return record


def build_task_result(
    *,
    task_id: str,
    attempt_id: str,
    route: str,
    seed: int,
    world_size: int,
    physical_gpu_ids: tuple[int, ...],
    source_manifest_sha256: str,
    data_sha256: str,
    parity: PairedRouteFacts,
    epochs: int,
    steps: int,
    warmup_steps: int,
    steady_samples_per_second: float,
    step_latency_p50_ms: float,
    step_latency_p95_ms: float,
    epoch_time_s: tuple[float, ...],
    communication_time_s: float,
    qwd_time_s: float,
    refresh_time_s: float,
    communication_bytes: int,
    peak_memory_mib: float,
    gpu_telemetry: tuple[Mapping[str, object], ...],
    loss_trajectory: tuple[float, ...],
    validation_loss: float,
    rank_gaps: tuple[float, ...],
    decision_counts: Mapping[str, int],
    failure_facts: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Build and freshly validate one completed-task schema-v1 result."""
    if type(parity) is not PairedRouteFacts:
        raise ValueError("parity must be exact PairedRouteFacts")
    precision, enabled, initial_scale = parity.amp_configuration
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "route": route,
        "seed": seed,
        "world_size": world_size,
        "physical_gpu_ids": list(physical_gpu_ids),
        "source_manifest_sha256": source_manifest_sha256,
        "data_sha256": data_sha256,
        "initial_parameter_sha256": parity.initial_parameter_sha256,
        "sampler_indices_sha256": canonical_sha256(parity.sampler_indices),
        "augmentation_rng_sha256": parity.augmentation_rng_sha256,
        "lr_schedule_sha256": canonical_sha256(parity.lr_schedule),
        "amp_configuration": {
            "precision": precision,
            "enabled": enabled,
            "initial_scale": initial_scale,
        },
        "batch_size_per_rank": parity.batch_size,
        "global_batch_size": parity.batch_size * world_size,
        "model_parameter_count": parity.model_parameter_count,
        "epochs": epochs,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "steady_samples_per_second": steady_samples_per_second,
        "step_latency_p50_ms": step_latency_p50_ms,
        "step_latency_p95_ms": step_latency_p95_ms,
        "epoch_time_s": list(epoch_time_s),
        "communication_time_s": communication_time_s,
        "qwd_time_s": qwd_time_s,
        "refresh_time_s": refresh_time_s,
        "communication_bytes": communication_bytes,
        "peak_memory_mib": peak_memory_mib,
        "gpu_telemetry": [dict(item) for item in gpu_telemetry],
        "loss_trajectory": list(loss_trajectory),
        "validation_loss": validation_loss,
        "rank_gaps": list(rank_gaps),
        "decision_counts": dict(decision_counts),
        "failure_facts": [dict(item) for item in failure_facts],
    }
    return validate_task_result(result)


def validate_task_result(value: object) -> dict[str, object]:
    """Freshly validate one exact completed-task schema-v1 result."""
    result = _require_exact_dict(value, _TASK_FIELDS, "task result")
    _require_schema_identity(result)
    _require_nonempty_str(result["task_id"], "task_id")
    _require_nonempty_str(result["attempt_id"], "attempt_id")
    _require_route(result["route"])
    _require_seed(result["seed"])
    _require_positive_int(result["world_size"], "world_size")
    _require_exact_int_list(result["physical_gpu_ids"], "physical_gpu_ids")
    if len(result["physical_gpu_ids"]) != result["world_size"]:
        raise ValueError("physical_gpu_ids must match world_size")
    for field in (
        "source_manifest_sha256",
        "data_sha256",
        "initial_parameter_sha256",
        "sampler_indices_sha256",
        "augmentation_rng_sha256",
        "lr_schedule_sha256",
    ):
        _require_sha256(result[field], field)
    amp = _require_exact_dict(result["amp_configuration"], _AMP_FIELDS, "amp")
    if amp["precision"] != "fp16" or type(amp["enabled"]) is not bool:
        raise ValueError("amp_configuration is invalid")
    _require_nonnegative_float(amp["initial_scale"], "initial_scale")
    for field in (
        "batch_size_per_rank",
        "global_batch_size",
        "model_parameter_count",
        "epochs",
    ):
        _require_positive_int(result[field], field)
    if result["global_batch_size"] != (
        result["batch_size_per_rank"] * result["world_size"]
    ):
        raise ValueError("global_batch_size is inconsistent")
    _require_positive_int(result["steps"], "steps")
    for field in ("warmup_steps", "communication_bytes"):
        _require_nonnegative_int(result[field], field)
    for field in (
        "steady_samples_per_second",
        "step_latency_p50_ms",
        "step_latency_p95_ms",
        "communication_time_s",
        "qwd_time_s",
        "refresh_time_s",
        "peak_memory_mib",
        "validation_loss",
    ):
        _require_nonnegative_float(result[field], field)
    for field in ("epoch_time_s", "loss_trajectory", "rank_gaps"):
        _require_nonnegative_float_list(result[field], field)
    telemetry = result["gpu_telemetry"]
    if type(telemetry) is not list:
        raise ValueError("gpu_telemetry must be an exact list")
    for item in telemetry:
        value = _require_exact_dict(item, _GPU_TELEMETRY_FIELDS, "gpu_telemetry")
        _require_nonnegative_int(value["gpu"], "gpu")
        for field in _GPU_TELEMETRY_FIELDS - {"gpu"}:
            _require_nonnegative_float(value[field], field)
    if type(result["decision_counts"]) is not dict or not all(
        type(key) is str and bool(key) and type(count) is int and count >= 0
        for key, count in result["decision_counts"].items()
    ):
        raise ValueError("decision_counts is invalid")
    failure_facts = result["failure_facts"]
    if type(failure_facts) is not list:
        raise ValueError("failure_facts must be an exact list")
    for item in failure_facts:
        value = _require_exact_dict(item, _FAILURE_FACT_FIELDS, "failure_facts")
        for field in ("phase", "category", "message"):
            _require_nonempty_str(value[field], field)
        _require_nonnegative_int(value["rank"], "rank")
        _require_nonnegative_int(value["step"], "step")
        if type(value["recoverable"]) is not bool:
            raise ValueError("recoverable must be an exact bool")
    steps = result["steps"]
    if result["warmup_steps"] >= steps:
        raise ValueError("warmup_steps must be less than steps")
    if len(result["loss_trajectory"]) != steps or len(result["rank_gaps"]) != steps:
        raise ValueError("steps must match loss_trajectory and rank_gaps")
    if sum(result["decision_counts"].values()) != steps:
        raise ValueError("decision_counts must sum to steps")
    if len(result["epoch_time_s"]) != result["epochs"]:
        raise ValueError("epoch_time_s must match epochs")
    return result


def _require_schema_identity(value: dict[str, object]) -> None:
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be exact integer 1")


def _require_exact_dict(
    value: object,
    fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _require_route(value: object) -> None:
    if type(value) is not str or value not in ROUTES:
        raise ValueError("route is invalid")


def _require_seed(value: object) -> None:
    if type(value) is not int or value not in PAIRED_SEEDS:
        raise ValueError("seed is invalid")


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _require_nonempty_str(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")


def _require_positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact integer")


def _require_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative exact integer")


def _require_exact_int_tuple(value: object, name: str) -> None:
    if type(value) is not tuple or not all(type(item) is int for item in value):
        raise ValueError(f"{name} must be an exact integer tuple")


def _require_exact_int_list(value: object, name: str) -> None:
    if type(value) is not list or not all(type(item) is int for item in value):
        raise ValueError(f"{name} must be an exact integer list")


def _is_nonnegative_finite_float(value: object) -> bool:
    return type(value) is float and isfinite(value) and value >= 0.0


def _require_finite_float(value: object, name: str) -> None:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")


def _require_nonnegative_float(value: object, name: str) -> None:
    if not _is_nonnegative_finite_float(value):
        raise ValueError(f"{name} must be a non-negative finite exact float")


def _require_nonnegative_float_list(value: object, name: str) -> None:
    if type(value) is not list or not all(
        _is_nonnegative_finite_float(item) for item in value
    ):
        raise ValueError(f"{name} must be a list of non-negative exact floats")


__all__ = [
    "DEFAULT_PSI_SOURCE",
    "PAIRED_SEEDS",
    "ROUTES",
    "PairedRouteFacts",
    "RSAGQWDTransaction",
    "ResumeFacts",
    "SCHEMA_VERSION",
    "StepTiming",
    "assert_paired_route_parity",
    "assert_resume_matches",
    "build_engine",
    "build_step_record",
    "build_task_result",
    "canonical_sha256",
    "parse_args",
    "validate_step_record",
    "validate_task_result",
]
