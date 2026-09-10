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
from tests.benchmarks.psi_training_runtime import validate_training_protocol

ROUTES = ("native", "cag", "rsag_qwd")
PAIRED_SEEDS = (20260821, 20260822, 20260823)
SCHEMA_VERSION = 1
WINDOW_SCHEMA_VERSION = 2
TASK_SCHEMA_VERSION = 2
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
        "audit_performed",
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
        "rank_devices",
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
        "epoch_core_time_s",
        "timing_breakdown",
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
_WINDOW_TASK_FIELDS = frozenset({
    "training_loop_windows", "steady_training_loop_window_s",
    "steady_training_loop_samples_per_second", "window_observability",
})
_RANK_DEVICE_FIELDS = frozenset(
    {
        "hostname",
        "node_rank",
        "global_rank",
        "local_rank",
        "visible_device",
        "physical_gpu_index",
        "gpu_uuid",
        "pci_bus_id",
    }
)
_TASK_TIMING_FIELDS = frozenset(
    {
        "startup_s",
        "data_s",
        "core_train_s",
        "validation_s",
        "checkpoint_s",
        "quality_audit_s",
        "report_s",
        "other_s",
        "process_wall_s",
    }
)
_AMP_FIELDS = frozenset(
    {
        "precision",
        "enabled",
        "initial_scale",
        "effective_start_scale",
        "growth_interval",
        "growth_factor",
        "backoff_factor",
    }
)
_GPU_TELEMETRY_FIELDS = frozenset(
    {
        "global_rank",
        "hostname",
        "gpu_uuid",
        "pci_bus_id",
        "physical_gpu_index",
        "sample_count",
        "window_s",
        "utilization_mean",
        "utilization_p50",
        "utilization_p95",
        "utilization_max",
        "memory_used_mib_mean",
        "memory_used_mib_p50",
        "memory_used_mib_p95",
        "memory_used_mib_max",
        "temperature_c_mean",
        "temperature_c_p95",
        "temperature_c_max",
        "sm_clock_mhz_mean",
        "sm_clock_mhz_p95",
        "sm_clock_mhz_max",
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
    amp_configuration: tuple[str, bool, float, float, int, float, float]
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
            or len(self.amp_configuration) != 7
            or type(self.amp_configuration[0]) is not str
            or self.amp_configuration[0] != "fp16"
            or type(self.amp_configuration[1]) is not bool
            or self.amp_configuration[1] is not True
            or not _is_nonnegative_finite_float(self.amp_configuration[2])
            or self.amp_configuration[2] <= 0.0
            or not _is_nonnegative_finite_float(self.amp_configuration[3])
            or self.amp_configuration[3] <= 0.0
            or type(self.amp_configuration[4]) is not int
            or self.amp_configuration[4] <= 0
            or not _is_nonnegative_finite_float(self.amp_configuration[5])
            or self.amp_configuration[5] <= 1.0
            or not _is_nonnegative_finite_float(self.amp_configuration[6])
            or not 0.0 < self.amp_configuration[6] < 1.0
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
    next_batch_sha256: str
    next_augmentation_sha256: str
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
        _require_sha256(self.next_batch_sha256, "next_batch_sha256")
        _require_sha256(
            self.next_augmentation_sha256,
            "next_augmentation_sha256",
        )
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
        "--model-precision", choices=("fp16", "fp32"), default="fp16",
        help="fp32: separate conventional AMP reference, native standard DDP only",
    )
    parser.add_argument(
        "--timing-mode", choices=("diagnostic", "production", "window"), default="diagnostic",
        help="window: boundary-synchronized training-loop wall timing; production: synchronized whole-step wall timing",
    )
    parser.add_argument(
        "--native-ddp-mode",
        choices=("standard", "diagnostic"),
        default="standard",
        help="standard: unmodified asynchronous DDP reducer; diagnostic: synchronous single bucket",
    )
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
    parser.add_argument(
        "--rsag-parameter-route",
        choices=("qwd_group64_refresh100", "all_refresh_fp32"),
        default="qwd_group64_refresh100",
        help="explicit RSAG parameter publication policy; all_refresh_fp32 is diagnostic only",
    )
    parser.add_argument(
        "--rsag-gradient-route",
        choices=("reduced_shard_int8_group64_ef", "reduced_shard_native_fp32"),
        default="reduced_shard_int8_group64_ef",
        help="explicit RSAG gradient reduction policy; native_fp32 is diagnostic only",
    )
    parser.add_argument("--data-mode", choices=("legacy", "deterministic"), default="legacy")
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--loader-prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp-initial-scale", type=float, default=1024.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--smoke-midpoint", type=int, default=25)
    parser.add_argument("--inject-overflow-step", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--cpu-affinity-map", default="")
    parser.add_argument("--nccl-channels", type=int, default=0)
    parser.add_argument(
        "--quality-audit-mode",
        choices=("production", "full"),
        default="production",
    )
    return parser.parse_args(argv)


def quality_audit_steps(total_steps: int) -> tuple[int, ...]:
    """Return deterministic one-based production audit checkpoints."""
    _require_positive_int(total_steps, "total_steps")
    return tuple(sorted({1, (total_steps + 1) // 2, total_steps}))


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
    loss: float | None,
    amp_scale: float,
    learning_rate: float,
    model_sha256: str | None,
    rank_parameter_gap: float | None,
    optimizer_step: int,
    finite: bool,
    audit_performed: bool = True,
    defer_loss_validation: bool = False,
    window_timing: bool = False,
) -> dict[str, object]:
    """Build and freshly validate one raw per-step schema-v1 row."""
    record: dict[str, object] = {
        "schema_version": WINDOW_SCHEMA_VERSION if window_timing else SCHEMA_VERSION,
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
            "audit_performed": audit_performed,
        },
    }
    if window_timing:
        record["timing"] = {field: None for field in _TIMING_FIELDS}
        record["communication"]["qwd_s"] = None
        record["communication"]["refresh_s"] = None
    if defer_loss_validation:
        if not window_timing:
            raise ValueError("deferred loss is valid only for window raw schema")
        if loss is None:
            record["quality"]["loss"] = 0.0
            validate_step_record(record)
            record["quality"]["loss"] = None
            return record
        return validate_step_record(record)
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
    if record["schema_version"] == WINDOW_SCHEMA_VERSION:
        if any(timing[field] is not None for field in _TIMING_FIELDS):
            raise ValueError("window step timing must be explicitly unavailable")
    else:
        for field in _TIMING_FIELDS:
            _require_nonnegative_float(timing[field], field)
        measured = timing["forward_s"] + timing["backward_s"] + timing["update_s"] + timing["communication_s"]
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
    for field in ("qwd_s", "refresh_s"):
        if record["schema_version"] == WINDOW_SCHEMA_VERSION:
            if communication[field] is not None: raise ValueError("window phase timing must be null")
        else: _require_nonnegative_float(communication[field], field)
    _require_nonempty_str(communication["decision"], "decision")

    quality = _require_exact_dict(record["quality"], _QUALITY_FIELDS, "quality")
    _require_finite_float(quality["loss"], "loss")
    _require_nonnegative_float(quality["amp_scale"], "amp_scale")
    _require_nonnegative_float(quality["learning_rate"], "learning_rate")
    if type(quality["audit_performed"]) is not bool:
        raise ValueError("audit_performed must be an exact bool")
    if quality["audit_performed"]:
        _require_sha256(quality["model_sha256"], "model_sha256")
        _require_nonnegative_float(quality["rank_parameter_gap"], "rank_parameter_gap")
    elif (
        quality["model_sha256"] is not None or quality["rank_parameter_gap"] is not None
    ):
        raise ValueError("unaudited quality must not publish audit facts")
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
    rank_devices: tuple[Mapping[str, object], ...],
    source_manifest_sha256: str,
    data_sha256: str,
    parity: PairedRouteFacts,
    epochs: int,
    steps: int,
    warmup_steps: int,
    steady_samples_per_second: float,
    step_latency_p50_ms: float,
    step_latency_p95_ms: float,
    epoch_core_time_s: tuple[float, ...],
    timing_breakdown: Mapping[str, float],
    communication_time_s: float,
    qwd_time_s: float,
    refresh_time_s: float,
    communication_bytes: int,
    peak_memory_mib: float,
    gpu_telemetry: tuple[Mapping[str, object], ...],
    loss_trajectory: tuple[float, ...],
    validation_loss: float,
    rank_gaps: tuple[float | None, ...],
    decision_counts: Mapping[str, int],
    failure_facts: tuple[Mapping[str, object], ...],
    execution_protocol: Mapping[str, object] | None = None,
    training_loop_windows: tuple[Mapping[str, object], ...] | None = None,
    local_record_samples: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Build legacy schema-v2 or corrected, protocol-bound schema-v3 results."""
    if type(parity) is not PairedRouteFacts:
        raise ValueError("parity must be exact PairedRouteFacts")
    (
        precision,
        enabled,
        initial_scale,
        effective_start_scale,
        growth_interval,
        growth_factor,
        backoff_factor,
    ) = parity.amp_configuration
    result: dict[str, object] = {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "route": route,
        "seed": seed,
        "world_size": world_size,
        "physical_gpu_ids": list(physical_gpu_ids),
        "rank_devices": [dict(item) for item in rank_devices],
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
            "effective_start_scale": effective_start_scale,
            "growth_interval": growth_interval,
            "growth_factor": growth_factor,
            "backoff_factor": backoff_factor,
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
        "epoch_core_time_s": list(epoch_core_time_s),
        "timing_breakdown": dict(timing_breakdown),
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
    if execution_protocol is not None:
        result["schema_version"] = 4 if execution_protocol.get("timing_mode") == "window" else 3
        result["execution_protocol"] = dict(execution_protocol)
    if result["schema_version"] == 4:
        windows = [dict(item) for item in (training_loop_windows or ())]
        warmup_limit = min(warmup_steps, steps)
        steady = [item for item in windows if item["local_record_range"][0] >= warmup_limit]
        elapsed = sum(float(item["elapsed_wall_s"]) for item in steady)
        samples = sum(int(item["samples"]) for item in steady)
        result.update(
            steady_samples_per_second=None, step_latency_p50_ms=None,
            step_latency_p95_ms=None, epoch_core_time_s=None,
            communication_time_s=None, qwd_time_s=None, refresh_time_s=None,
            training_loop_windows=windows,
            steady_training_loop_window_s=float(elapsed) if elapsed > 0 else None,
            steady_training_loop_samples_per_second=float(samples * world_size / elapsed) if elapsed > 0 and samples > 0 else None,
            window_observability={"measurement":"training_loop_window_wall", "single_step_timing_available":False,
                                  "phase_breakdown_available":False, "losses_flushed":True,
                                  "start_epoch":windows[0]["epoch"] if windows else None,
                                  "start_global_step":windows[0]["global_step_range"][0] if windows else None,
                                  "local_record_samples":list(local_record_samples or ())},
        )
        result["timing_breakdown"]["data_s"] = None
        result["timing_breakdown"]["core_train_s"] = None
    return validate_task_result(result)


def validate_task_result(value: object) -> dict[str, object]:
    """Read legacy v2 without requalification; v3 requires corrected protocol."""
    fields = _TASK_FIELDS
    if type(value) is dict and value.get("schema_version") in {3, 4}:
        fields = fields | {"execution_protocol"}
    if type(value) is dict and value.get("schema_version") == 4:
        fields = fields | _WINDOW_TASK_FIELDS
    result = _require_exact_dict(value, fields, "task result")
    _require_task_schema_identity(result)
    if result["schema_version"] in {3, 4}:
        protocol = result["execution_protocol"]
        validate_training_protocol(protocol)
        if protocol["model_precision"] == "fp32" and result["route"] != "native":
            raise ValueError("FP32 model reference requires native route")
        if (
            protocol["rank"] != 0
            or protocol["world_size"] != result["world_size"]
            or protocol["batch_size_per_rank"] != result["batch_size_per_rank"]
        ):
            raise ValueError("execution protocol does not match result geometry")
        if result["schema_version"] == 3 and (protocol.get("version") != 3 or protocol.get("timing_mode") == "window"):
            raise ValueError("numeric result schema requires non-window protocol version 3")
    _require_nonempty_str(result["task_id"], "task_id")
    _require_nonempty_str(result["attempt_id"], "attempt_id")
    _require_route(result["route"])
    _require_seed(result["seed"])
    _require_positive_int(result["world_size"], "world_size")
    _require_exact_int_list(result["physical_gpu_ids"], "physical_gpu_ids")
    if len(result["physical_gpu_ids"]) != result["world_size"]:
        raise ValueError("physical_gpu_ids must match world_size")
    _validate_rank_devices(result)
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
    if (
        amp["precision"] != "fp16"
        or type(amp["enabled"]) is not bool
        or amp["enabled"] is not True
        or not _is_nonnegative_finite_float(amp["initial_scale"])
        or amp["initial_scale"] <= 0.0
        or not _is_nonnegative_finite_float(amp["effective_start_scale"])
        or amp["effective_start_scale"] <= 0.0
        or type(amp["growth_interval"]) is not int
        or amp["growth_interval"] <= 0
        or not _is_nonnegative_finite_float(amp["growth_factor"])
        or amp["growth_factor"] <= 1.0
        or not _is_nonnegative_finite_float(amp["backoff_factor"])
        or not 0.0 < amp["backoff_factor"] < 1.0
    ):
        raise ValueError("amp_configuration is invalid")
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
    numeric_summary_fields = (
        "steady_samples_per_second",
        "step_latency_p50_ms",
        "step_latency_p95_ms",
        "communication_time_s",
        "qwd_time_s",
        "refresh_time_s",
        "peak_memory_mib",
        "validation_loss",
    )
    for field in numeric_summary_fields:
        if result["schema_version"] == 4 and field in {"steady_samples_per_second","step_latency_p50_ms","step_latency_p95_ms","communication_time_s","qwd_time_s","refresh_time_s"}:
            if result[field] is not None: raise ValueError("window step/core metric must be null")
        else: _require_nonnegative_float(result[field], field)
    if result["schema_version"] == 4:
        if result["epoch_core_time_s"] is not None: raise ValueError("window core time must be null")
        _validate_window_result(result)
    else:
        _require_nonnegative_float_list(result["epoch_core_time_s"], "epoch_core_time_s")
    _require_nonnegative_float_list(result["loss_trajectory"], "loss_trajectory")
    _require_optional_nonnegative_float_list(result["rank_gaps"], "rank_gaps")
    if not any(value is not None for value in result["rank_gaps"]):
        raise ValueError("rank_gaps must contain audited values")
    if result["rank_gaps"][-1] is None:
        raise ValueError("rank_gaps must audit the final step")
    _validate_task_timing(result)
    telemetry = result["gpu_telemetry"]
    if type(telemetry) is not list:
        raise ValueError("gpu_telemetry must be an exact list")
    for item in telemetry:
        value = _require_exact_dict(item, _GPU_TELEMETRY_FIELDS, "gpu_telemetry")
        for field in ("global_rank", "physical_gpu_index"):
            _require_nonnegative_int(value[field], field)
        _require_positive_int(value["sample_count"], "sample_count")
        for field in ("hostname", "gpu_uuid", "pci_bus_id"):
            _require_nonempty_str(value[field], field)
        for field in _GPU_TELEMETRY_FIELDS - {
            "global_rank",
            "hostname",
            "gpu_uuid",
            "pci_bus_id",
            "physical_gpu_index",
            "sample_count",
        }:
            _require_nonnegative_float(value[field], field)
    devices_by_rank = {item["global_rank"]: item for item in result["rank_devices"]}
    telemetry_by_rank = {item["global_rank"]: item for item in telemetry}
    if len(telemetry_by_rank) != result["world_size"]:
        raise ValueError("gpu_telemetry must contain every global rank")
    for rank, item in telemetry_by_rank.items():
        identity = devices_by_rank.get(rank)
        if identity is None or any(
            item[field] != identity[field]
            for field in (
                "hostname",
                "gpu_uuid",
                "pci_bus_id",
                "physical_gpu_index",
            )
        ):
            raise ValueError("gpu_telemetry must match rank device identity")
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
    if result["schema_version"] != 4 and result["warmup_steps"] >= steps:
        raise ValueError("warmup_steps must be less than steps")
    if len(result["loss_trajectory"]) != steps or len(result["rank_gaps"]) != steps:
        raise ValueError("steps must match loss_trajectory and rank_gaps")
    if sum(result["decision_counts"].values()) != steps:
        raise ValueError("decision_counts must sum to steps")
    if result["schema_version"] != 4 and len(result["epoch_core_time_s"]) != result["epochs"]:
        raise ValueError("epoch_core_time_s must match epochs")
    return result


def _validate_rank_devices(result: dict[str, object]) -> None:
    devices = result["rank_devices"]
    world_size = result["world_size"]
    if type(devices) is not list or len(devices) != world_size:
        raise ValueError("rank device identity must match world_size")
    validated = [
        _require_exact_dict(item, _RANK_DEVICE_FIELDS, "rank device identity")
        for item in devices
    ]
    for item in validated:
        for field in ("hostname", "visible_device", "gpu_uuid", "pci_bus_id"):
            _require_nonempty_str(item[field], field)
        for field in (
            "node_rank",
            "global_rank",
            "local_rank",
            "physical_gpu_index",
        ):
            _require_nonnegative_int(item[field], field)
    global_ranks = [item["global_rank"] for item in validated]
    if sorted(global_ranks) != list(range(world_size)):
        raise ValueError("rank device identity global ranks are invalid")
    identities = [item["gpu_uuid"] for item in validated]
    if len(set(identities)) != world_size:
        raise ValueError("rank device identity GPU UUIDs must be unique")
    rank_slots = [(item["hostname"], item["local_rank"]) for item in validated]
    if len(set(rank_slots)) != world_size:
        raise ValueError("rank device identity local rank slots must be unique")
    by_rank = sorted(validated, key=lambda item: item["global_rank"])
    physical = [item["physical_gpu_index"] for item in by_rank]
    if physical != result["physical_gpu_ids"]:
        raise ValueError("rank device identity disagrees with physical GPU IDs")


def _validate_task_timing(result: dict[str, object]) -> None:
    timing = _require_exact_dict(
        result["timing_breakdown"],
        _TASK_TIMING_FIELDS,
        "timing_breakdown",
    )
    if result["schema_version"] == 4:
        if timing["data_s"] is not None or timing["core_train_s"] is not None:
            raise ValueError("overlapping window components must be null")
        for field in _TASK_TIMING_FIELDS - {"data_s", "core_train_s"}:
            _require_nonnegative_float(timing[field], field)
        return
    for field in _TASK_TIMING_FIELDS: _require_nonnegative_float(timing[field], field)
    core_total = sum(result["epoch_core_time_s"])
    if not _close_float(timing["core_train_s"], core_total):
        raise ValueError("timing_breakdown core train time is inconsistent")
    components = sum(
        timing[field] for field in _TASK_TIMING_FIELDS - {"process_wall_s"}
    )
    if not _close_float(timing["process_wall_s"], components):
        raise ValueError("timing_breakdown does not close to process wall")


def _close_float(left: object, right: object) -> bool:
    if type(left) is not float or type(right) is not float:
        return False
    tolerance = max(1.0e-9, abs(right) * 1.0e-9)
    return abs(left - right) <= tolerance


def _require_schema_identity(value: dict[str, object]) -> None:
    if type(value["schema_version"]) is not int or value["schema_version"] not in {1, 2}:
        raise ValueError("schema_version must be exact integer 1 or 2")


def _require_task_schema_identity(value: dict[str, object]) -> None:
    if type(value["schema_version"]) is not int or value["schema_version"] not in {
        TASK_SCHEMA_VERSION,
        3,
        4,
    }:
        raise ValueError("task schema_version must be exact integer 2, 3 or 4")


def _validate_window_result(result: dict[str, object]) -> None:
    protocol = result["execution_protocol"]
    if protocol.get("version") != 4 or protocol.get("timing_mode") != "window":
        raise ValueError("window result requires protocol version 4")
    observability = _require_exact_dict(result["window_observability"], frozenset({"measurement","single_step_timing_available","phase_breakdown_available","losses_flushed","start_epoch","start_global_step","local_record_samples"}), "window observability")
    if (observability["measurement"] != "training_loop_window_wall"
            or observability["single_step_timing_available"] is not False
            or observability["phase_breakdown_available"] is not False
            or observability["losses_flushed"] is not True):
        raise ValueError("window observability is inconsistent")
    windows = result["training_loop_windows"]
    if type(windows) is not list or not windows: raise ValueError("training_loop_windows must be a nonempty list")
    samples_by_record = observability["local_record_samples"]
    if type(samples_by_record) is not list or len(samples_by_record) != result["steps"] or any(type(value) is not int or value <= 0 or value > result["batch_size_per_rank"] for value in samples_by_record):
        raise ValueError("local record samples are invalid")
    if type(observability["start_epoch"]) is not int or observability["start_epoch"] < 0 or type(observability["start_global_step"]) is not int or observability["start_global_step"] < 0:
        raise ValueError("window resumed start metadata is invalid")
    prior_local_end = 0
    prior_global_end = observability["start_global_step"]
    prior_epoch = observability["start_epoch"]
    for index, item in enumerate(windows):
        window = _require_exact_dict(item, frozenset({"epoch","kind","local_record_range","global_step_range","samples","elapsed_wall_s"}), "training loop window")
        if window["kind"] not in {"warmup","epoch","training_stop"}: raise ValueError("invalid window kind")
        _require_nonnegative_int(window["epoch"], "window epoch"); _require_positive_int(window["samples"], "window samples"); _require_nonnegative_float(window["elapsed_wall_s"], "window elapsed")
        if window["elapsed_wall_s"] <= 0.0: raise ValueError("window elapsed must be positive")
        for name in ("local_record_range", "global_step_range"):
            values = window[name]
            if type(values) is not list or len(values) != 2 or any(type(value) is not int or value < 0 for value in values) or values[1] <= values[0]:
                raise ValueError(f"{name} is invalid")
        if window["local_record_range"][0] != prior_local_end:
            raise ValueError("window local record coverage is not contiguous")
        local_start, local_end = window["local_record_range"]
        global_start, global_end = window["global_step_range"]
        if global_start != prior_global_end or global_end - global_start != local_end - local_start:
            raise ValueError("window global/local record geometry is inconsistent")
        if window["samples"] != sum(samples_by_record[local_start:local_end]):
            raise ValueError("window samples disagree with record geometry")
        if window["epoch"] not in {prior_epoch, prior_epoch + 1}:
            raise ValueError("window epoch progression is invalid")
        if window["epoch"] == prior_epoch + 1 and (index == 0 or windows[index - 1]["kind"] not in {"epoch", "warmup"}):
            raise ValueError("window epoch transition lacks an epoch boundary")
        warmup_limit = min(result["warmup_steps"], result["steps"])
        if local_start < warmup_limit < local_end:
            raise ValueError("window crosses the exact local warmup boundary")
        if window["kind"] == "warmup" and local_end != warmup_limit:
            raise ValueError("warmup boundary window ends at the wrong local record")
        prior_local_end = window["local_record_range"][1]
        prior_global_end = global_end; prior_epoch = window["epoch"]
    if prior_local_end != result["steps"]:
        raise ValueError("window coverage must contain every local record")
    if len({window["epoch"] for window in windows}) != result["epochs"]:
        raise ValueError("window epochs disagree with result")
    if any(window["kind"] == "training_stop" for window in windows[:-1]):
        raise ValueError("window terminal boundary is invalid")
    for index, window in enumerate(windows[:-1]):
        following = windows[index + 1]
        if following["epoch"] == window["epoch"]:
            if window["kind"] != "warmup":
                raise ValueError("unexpected within-epoch window split")
        elif following["epoch"] != window["epoch"] + 1 or window["kind"] not in {"epoch", "warmup"}:
            raise ValueError("window epoch boundary sequence is invalid")
    warmup_limit = min(result["warmup_steps"], result["steps"])
    steady = [window for window in windows if window["local_record_range"][0] >= warmup_limit]
    elapsed = sum(window["elapsed_wall_s"] for window in steady)
    samples = sum(window["samples"] for window in steady)
    expected_elapsed = float(elapsed) if elapsed > 0 else None
    expected_rate = float(samples * result["world_size"] / elapsed) if elapsed > 0 and samples > 0 else None
    for field, expected in (("steady_training_loop_window_s", expected_elapsed),
                            ("steady_training_loop_samples_per_second", expected_rate)):
        observed = result[field]
        if expected is None:
            if observed is not None: raise ValueError(f"{field} must be null without steady coverage")
        elif type(observed) is not float or not isfinite(observed) or observed <= 0.0:
            raise ValueError(f"{field} must be an exact finite positive float")
    if result["steady_training_loop_window_s"] != expected_elapsed or result["steady_training_loop_samples_per_second"] != expected_rate:
        raise ValueError("steady training-loop window summary is inconsistent")


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


def _require_optional_nonnegative_float_list(
    value: object,
    name: str,
) -> None:
    if type(value) is not list or not all(
        item is None or _is_nonnegative_finite_float(item) for item in value
    ):
        raise ValueError(f"{name} must contain optional non-negative exact floats")


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
    "quality_audit_steps",
    "validate_step_record",
    "validate_task_result",
]
