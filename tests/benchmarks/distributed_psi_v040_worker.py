"""One distributed worker for Native, CAG, and RSAG/qWD PSI training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from itertools import chain, islice
import json
from math import isfinite
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace
from lowbit_comm.backends.cuda.placement import (
    apply_cuda_process_placement,
    parse_cuda_process_placement,
)
from lowbit_comm.experimental.rsag import (
    QWDSchedule,
    ShardLayout,
    ShardedAdamW,
    _create_rsag_qwd_plans,
)
from tests.benchmarks.psi_v040_training import (
    PairedRouteFacts,
    RSAGQWDTransaction,
    ResumeFacts,
    StepTiming,
    assert_resume_matches,
    build_engine,
    build_step_record,
    build_task_result,
    canonical_sha256,
    parse_args,
    quality_audit_steps,
    validate_step_record,
)
from tests.benchmarks.psi_training_runtime import (
    BatchWaitTimer,
    FP32MasterWeights,
    PhaseTimer,
    TrainingLoopWindowObserver,
    RankBatchSampler,
    data_loader_seed,
    execution_counters,
    loader_configuration,
    measurement_observability,
    norm_clip_coefficient,
    training_protocol,
    validate_training_protocol,
)

_CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache", "__MACOSX"})
_DDP_BUCKET_WARMUP_BACKWARDS = 2
_DEPRECATED_PSI_COMMUNICATION_FILES = (
    "psi_policy/communication/ccdl_ddp.py",
    "psi_policy/communication/ccdl_sharded_adamw.py",
    "psi_policy/communication/ccdl_sharded_adamw.py.pre_qwd",
    "psi_policy/communication/legacy_torch_sharded_adamw.py",
)
_DEPRECATED_PSI_COMMUNICATION_IMPORTS = (
    "psi_policy.communication.ccdl_ddp",
    "psi_policy.communication.ccdl_sharded_adamw",
)
_LOCKED_OVERRIDE_KEYS = frozenset(
    {
        "communication.enabled",
        "logging.mode",
        "notifications.feishu.enabled",
        "train_dataloader.batch_size",
        "train_dataloader.num_workers",
        "train_dataloader.persistent_workers",
        "training.gradient_accumulate_every",
        "training.num_epochs",
        "training.seed",
        "training.use_ema",
        "val_dataloader.num_workers",
        "val_dataloader.persistent_workers",
    }
)


@dataclass(slots=True)
class _AmpScaleState:
    value: float


def source_tree_manifest(root: str | Path) -> dict[str, object]:
    """Return a stable per-file manifest without mutating the PSI tree."""
    source = Path(root)
    if not source.is_dir():
        raise ValueError("PSI source root must be an existing directory")
    files: list[tuple[str, str]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or any(part in _CACHE_PARTS for part in relative.parts)
            or path.name == ".DS_Store"
            or path.suffix == ".pyc"
        ):
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        files.append((relative.as_posix(), digest))
    frozen_files = tuple(files)
    return {
        "manifest_sha256": canonical_sha256(frozen_files),
        "file_count": len(frozen_files),
        "files": frozen_files,
    }


def summarize_step_records(
    records: tuple[dict[str, object], ...],
    *,
    warmup_steps: int,
    batch_size_per_rank: int,
    world_size: int,
) -> dict[str, object]:
    """Aggregate exact raw rows while excluding warmup from steady metrics."""
    if type(records) is not tuple or not records:
        raise ValueError("records must be a non-empty exact tuple")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative exact integer")
    if type(batch_size_per_rank) is not int or batch_size_per_rank <= 0:
        raise ValueError("batch_size_per_rank must be positive")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be positive")
    validated = tuple(validate_step_record(record) for record in records)
    steady = validated[warmup_steps:]
    if not steady:
        raise ValueError("warmup_steps leaves no steady records")
    latencies = sorted(float(record["timing"]["measured_s"]) for record in steady)
    steady_seconds = sum(latencies)
    if steady_seconds <= 0.0:
        raise ValueError("steady measured time must be positive")
    samples = len(steady) * batch_size_per_rank * world_size
    communications = tuple(record["communication"] for record in validated)
    qualities = tuple(record["quality"] for record in validated)
    counts = Counter(str(value["decision"]) for value in communications)
    return {
        "steady_samples_per_second": samples / steady_seconds,
        "step_latency_p50_ms": _percentile(latencies, 0.50) * 1000.0,
        "step_latency_p95_ms": _percentile(latencies, 0.95) * 1000.0,
        "communication_time_s": sum(
            float(record["timing"]["communication_s"]) for record in validated
        ),
        "qwd_time_s": sum(float(value["qwd_s"]) for value in communications),
        "refresh_time_s": sum(float(value["refresh_s"]) for value in communications),
        "communication_bytes": sum(int(value["bytes"]) for value in communications),
        "loss_trajectory": tuple(float(value["loss"]) for value in qualities),
        "rank_gaps": tuple(value["rank_parameter_gap"] for value in qualities),
        "decision_counts": dict(counts),
    }


def summarize_window_records(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Aggregate only non-timing facts; window wall metrics live separately."""
    validated = tuple(validate_step_record(record) for record in records)
    communications = tuple(record["communication"] for record in validated)
    qualities = tuple(record["quality"] for record in validated)
    return {
        "communication_time_s": 0.0,
        "qwd_time_s": 0.0,
        "refresh_time_s": 0.0,
        "communication_bytes": sum(int(value["bytes"]) for value in communications),
        "loss_trajectory": tuple(float(value["loss"]) for value in qualities),
        "rank_gaps": tuple(value["rank_parameter_gap"] for value in qualities),
        "decision_counts": dict(Counter(str(value["decision"]) for value in communications)),
    }


def window_audit_required(*, quality_audit_mode: str, global_step: int,
                          audit_steps: frozenset[int], final_batch: bool,
                          midpoint: bool, pending_resume: bool,
                          resume_oracle: bool) -> bool:
    """The single production-used decision for immediate window loss reads."""
    return (quality_audit_mode == "full" or global_step in audit_steps or final_batch
            or midpoint or pending_resume or resume_oracle)


def validate_window_rank_coverage(gathered_execution: list[object]) -> None:
    coverage = [[{key: value for key, value in window.items() if key != "elapsed_wall_s"}
                 for window in item["training_loop_windows"]] for item in gathered_execution]
    if not coverage or any(value != coverage[0] for value in coverage[1:]):
        raise RuntimeError("rank training-loop window coverage mismatch")


def observed_process_wall(timing_mode: str, observed: float, accounted: float) -> float:
    return observed if timing_mode == "window" else max(observed, accounted)


def window_sidecar_fields(observer: TrainingLoopWindowObserver | None) -> dict[str, object]:
    return ({"training_loop_windows": list(observer.windows)}
            if observer is not None else {})


def commit_observed_step(*, timing_mode: str, observer: TrainingLoopWindowObserver | None,
                         records: list[dict[str, object]], loss: object,
                         audit_performed: bool, quality_audit, bookkeeping,
                         build_record, post_append,
                         epoch: int, global_step: int, samples: int,
                         final_batch: bool, training_stop: bool
                         ) -> tuple[dict[str, object], dict[str, object]]:
    """Production seam preserving the historical observation/failure ordering."""
    detached_loss = loss.detach()
    loss_value = (None if timing_mode == "window" and not audit_performed
                  else float(detached_loss))
    audit = quality_audit()
    bookkeeping(loss_value, audit)
    record = build_record(loss_value, audit)
    records.append(record)
    post_append(record, loss_value, audit)
    if observer is not None:
        observer.observe(loss=detached_loss, record=record, audit=audit_performed,
                         epoch=epoch, local_record_end=len(records),
                         global_step_end=global_step, samples=samples)
        if not observer.is_open and not training_stop and not final_batch:
            observer.begin(epoch=epoch, local_record_start=len(records),
                           global_step_start=global_step,
                           boundary_already_synchronized=True)
    return record, audit


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


_PHASE_TIMER = PhaseTimer("diagnostic", lambda: _torch().cuda)


def _cuda_timed(action: Callable[[], object]) -> tuple[object, float]:
    return _PHASE_TIMER.measure(action)


@dataclass(slots=True)
class _HookTelemetry:
    communication_s: float = 0.0
    communication_bytes: int = 0
    plans: dict[tuple[object, ...], object] = field(default_factory=dict)
    feedback_snapshots: dict[tuple[object, ...], object | None] = field(
        default_factory=dict
    )
    pending_feedback: dict[tuple[object, ...], dict[str, object]] = field(
        default_factory=dict
    )
    restoring: bool = False

    def add(self, elapsed_s: float, byte_count: int) -> None:
        self.communication_s += elapsed_s
        self.communication_bytes += byte_count

    def bind_plan(self, key: tuple[object, ...], plan: object) -> None:
        if key in self.plans:
            raise RuntimeError("CAG bucket plan identity was rebound")
        if self.restoring:
            if key not in self.pending_feedback:
                raise ValueError("CAG checkpoint bucket identity is inconsistent")
            _restore_plan_feedback(
                plan,
                self.pending_feedback.pop(key),
                expected_key=key,
                expected_device=_torch().device(str(key[-2]), key[-1]),
            )
        self.plans[key] = plan

    def snapshot_feedback(self, key: tuple[object, ...]) -> None:
        if key not in self.feedback_snapshots:
            self.feedback_snapshots[key] = _clone_plan_feedback(self.plans[key])

    def consume(self, *, commit_feedback: bool = True) -> tuple[float, int]:
        if not commit_feedback:
            for key, residual in self.feedback_snapshots.items():
                self.plans[key]._restore_committed_residual(residual)
        if self.restoring and self.pending_feedback:
            raise ValueError("CAG checkpoint is missing restored bucket plans")
        if self.restoring:
            self.restoring = False
        self.feedback_snapshots.clear()
        result = self.communication_s, self.communication_bytes
        self.communication_s = 0.0
        self.communication_bytes = 0
        return result

    def state_dict(self) -> dict[str, object]:
        if self.feedback_snapshots:
            raise RuntimeError("CAG feedback checkpoint requires a step boundary")
        if self.restoring:
            return {
                "plans": tuple(
                    self.pending_feedback[key] for key in sorted(self.pending_feedback)
                )
            }
        entries = []
        for key in sorted(self.plans):
            plan = self.plans[key]
            entries.append(
                {
                    "key": key,
                    "layout": _plan_layout_facts(plan),
                    "residual": _clone_plan_feedback(plan),
                }
            )
        return {"plans": tuple(entries)}

    def _audit_state(self) -> dict[str, object]:
        """Borrow committed residuals for one immediate synchronous hash."""
        if self.feedback_snapshots:
            raise RuntimeError("CAG feedback audit requires a step boundary")
        if self.restoring:
            return {
                "plans": tuple(
                    self.pending_feedback[key] for key in sorted(self.pending_feedback)
                )
            }
        return {
            "plans": tuple(
                {
                    "key": key,
                    "layout": _plan_layout_facts(self.plans[key]),
                    "residual": self.plans[key]._committed_residual,
                }
                for key in sorted(self.plans)
            )
        }

    def load_state_dict(self, state: object) -> None:
        _require_fields(state, {"plans"}, "CAG feedback state")
        entries = state["plans"]
        if type(entries) is not tuple:
            raise ValueError("CAG feedback plans must be an exact tuple")
        pending: dict[tuple[object, ...], dict[str, object]] = {}
        for entry in entries:
            _require_fields(entry, {"key", "layout", "residual"}, "CAG plan state")
            key = entry["key"]
            if type(key) is not tuple or key in pending:
                raise ValueError("CAG checkpoint bucket key is invalid")
            pending[key] = entry
        self.pending_feedback = pending
        self.restoring = bool(pending)

    def reset_after_warmup(self) -> None:
        if self.restoring or self.pending_feedback:
            raise RuntimeError("DDP warmup cannot replace checkpoint feedback")
        self.consume(commit_feedback=False)
        self.plans.clear()


class NativeUpdateEngine:
    """Full AdamW after exact DDP/NCCL mean-gradient synchronization."""

    route = "native"
    gradient_route = "ddp_nccl"
    parameter_route = "full_adamw"
    gradients_are_unscaled = False

    def __init__(
        self,
        *,
        model: object,
        optimizer: object,
        grad_clip: float,
        telemetry: _HookTelemetry,
        amp_scale: _AmpScaleState,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.master_weights = FP32MasterWeights(optimizer)
        self.grad_clip = grad_clip
        self.telemetry = telemetry
        self.amp_scale = amp_scale
        self.step_count = 0

    def step(self, scaler: object) -> dict[str, object]:
        torch = _torch()
        overflow, _, overflow_control_s, overflow_control_bytes = (
            _unscale_and_detect_overflow(
                tuple(self.model.parameters()),
                self.amp_scale.value,
                torch.distributed.group.WORLD,
                already_unscaled=self.gradients_are_unscaled,
            )
        )

        def update_optimizer() -> None:
            if not overflow:
                self.master_weights.prepare_gradients()
                torch.nn.utils.clip_grad_norm_(
                    self.master_weights.master_parameters,
                    self.grad_clip,
                )
                self.optimizer.step()
                self.master_weights.publish()

        _, update_s = _cuda_timed(update_optimizer)
        self.amp_scale.value = _advance_amp_scaler(scaler, overflow=overflow)
        skipped = overflow
        self.master_weights.zero_grad()
        if not skipped:
            self.step_count += 1
        communication_s, communication_bytes = self.telemetry.consume(
            commit_feedback=not skipped
        )
        communication_s += overflow_control_s
        communication_bytes += overflow_control_bytes
        return {
            "update_s": update_s,
            "communication_s": communication_s,
            "communication_bytes": communication_bytes,
            "qwd_s": 0.0,
            "refresh_s": 0.0,
            "decision": ("overflow_after_communication" if skipped else self.route),
            "skipped": skipped,
            "update_communication_s": overflow_control_s,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_weights": self.master_weights.state_dict(),
            "step_count": self.step_count,
        }

    def _audit_state(self) -> dict[str, object]:
        """Borrow optimizer tensors for one immediate synchronous hash."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_weights": self.master_weights.audit_state(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        _require_fields(
            state, {"optimizer", "master_weights", "step_count"}, "native state"
        )
        self.master_weights.validate_state_dict(state["master_weights"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.master_weights.load_state_dict(state["master_weights"])
        self.step_count = int(state["step_count"])


class CAGUpdateEngine(NativeUpdateEngine):
    """Full AdamW after v0.4 FullTensor transactional-EF CAG."""

    route = "cag"
    gradient_route = "fulltensor_int8_group64_ef"
    gradients_are_unscaled = True

    def step(self, scaler: object) -> dict[str, object]:
        result = super().step(scaler)
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            **super().state_dict(),
            "gradient_feedback": self.telemetry.state_dict(),
        }

    def _audit_state(self) -> dict[str, object]:
        """Borrow optimizer and EF tensors for immediate synchronous hashing."""
        return {
            **super()._audit_state(),
            "gradient_feedback": self.telemetry._audit_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fields = {"optimizer", "master_weights", "step_count", "gradient_feedback"}
        _require_fields(state, fields, "CAG state")
        super().load_state_dict(
            {key: state[key] for key in fields - {"gradient_feedback"}}
        )
        self.telemetry.load_state_dict(state["gradient_feedback"])


class RSAGQWDUpdateEngine:
    """ReducedShard AdamW and private qWD/refresh flat-model commit."""

    route = "rsag_qwd"
    gradient_route = "reduced_shard_int8_group64_ef"
    parameter_route = "qwd_group64_refresh100"

    def __init__(
        self,
        *,
        model: object,
        optimizer: object,
        grad_clip: float,
        rank: int,
        world_size: int,
        process_group: object,
        amp_scale: _AmpScaleState,
        parameter_route: str = "qwd_group64_refresh100",
        gradient_route: str = "reduced_shard_int8_group64_ef",
    ) -> None:
        torch = _torch()
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.amp_scale = amp_scale
        if type(parameter_route) is not str or parameter_route not in {
            "qwd_group64_refresh100",
            "all_refresh_fp32",
        }:
            raise ValueError("RSAG/qWD parameter route is invalid")
        if type(gradient_route) is not str or gradient_route not in {
            "reduced_shard_int8_group64_ef",
            "reduced_shard_native_fp32",
        }:
            raise ValueError("RSAG/qWD gradient route is invalid")
        self.parameter_route = parameter_route
        self.gradient_route = gradient_route
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not self.parameters:
            raise RuntimeError("RSAG/qWD requires trainable parameters")
        self.global_numel = sum(parameter.numel() for parameter in self.parameters)
        self.layout = ShardLayout.build(self.global_numel, world_size, rank)
        self.padded_model_numel = self.layout.padded_numel * world_size
        flat = self._flat_model()
        # Scratch input only: never aliases model, optimizer, or checkpoint state.
        # A step waits for gradient Work before this storage can be reused.
        self._flat_gradient = torch.empty(
            self.global_numel, device=flat.device, dtype=torch.float16,
        )
        padded = torch.zeros(
            self.layout.padded_numel,
            device=flat.device,
            dtype=torch.float32,
        )
        valid = self.layout.valid_numel
        if valid:
            padded[:valid].copy_(flat.narrow(0, self.layout.start, valid).float())
        groups = tuple(self.optimizer.param_groups)
        if not groups:
            raise RuntimeError("RSAG/qWD requires optimizer parameter groups")
        betas = tuple(float(value) for value in groups[0]["betas"])
        eps = float(groups[0].get("eps", 1.0e-8))
        if any(
            tuple(float(value) for value in group["betas"]) != betas for group in groups
        ):
            raise RuntimeError("RSAG/qWD requires identical AdamW betas")
        if any(float(group.get("eps", 1.0e-8)) != eps for group in groups):
            raise RuntimeError("RSAG/qWD requires identical AdamW epsilon")
        initial_learning_rates = tuple(
            float(group.get("initial_lr", group["lr"])) for group in groups
        )
        if len(set(initial_learning_rates)) != 1 or initial_learning_rates[0] <= 0.0:
            raise RuntimeError("RSAG/qWD requires one positive base learning rate")
        self.base_learning_rate = initial_learning_rates[0]
        self.sharded_optimizer = ShardedAdamW(
            self.layout,
            padded,
            learning_rate=self.base_learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )
        self.candidate_optimizer = ShardedAdamW(
            self.layout,
            padded,
            learning_rate=self.base_learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )
        self.weight_decay = self._weight_decay_shard()
        self.model_copy_flat = torch.zeros(
            self.padded_model_numel,
            device=flat.device,
            dtype=torch.float16,
        )
        self._copy_model_to_padded_flat()
        self.force_refresh = False
        self.schedule = QWDSchedule(
            refresh_interval=100,
            policy=(
                "all_refresh"
                if parameter_route == "all_refresh_fp32"
                else "interval100"
            ),
        )
        plans = _create_rsag_qwd_plans(
            process_group,
            global_numel=self.global_numel,
            rank=rank,
            world_size=world_size,
            gradient_route=gradient_route,
        )
        if plans.layout != self.layout:
            raise RuntimeError("packaged RSAG/qWD layout is inconsistent")
        self.gradient_plan = plans.gradient_plan
        self.qwd_plan = plans.qwd_plan
        self.qwd_gathered_payload_bytes = plans.qwd_gathered_payload_bytes
        self.fp32_gathered_bytes = plans.fp32_gathered_bytes

    @property
    def master(self) -> object:
        return self.sharded_optimizer.master

    @property
    def exp_avg(self) -> object:
        return self.sharded_optimizer.exp_avg

    @property
    def exp_avg_sq(self) -> object:
        return self.sharded_optimizer.exp_avg_sq

    @property
    def step_count(self) -> int:
        return self.sharded_optimizer.step_count

    def step(self, scaler: object) -> dict[str, object]:
        torch = _torch()
        gradients = tuple(parameter.grad for parameter in self.parameters)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("every RSAG/qWD parameter requires a gradient")
        overflow, _, overflow_control_s, overflow_control_bytes = (
            _unscale_and_detect_overflow(
                self.parameters,
                self.amp_scale.value,
                self.process_group,
            )
        )
        if overflow:
            self.optimizer.zero_grad(set_to_none=True)
            result = {
                "update_s": 0.0,
                "communication_s": overflow_control_s,
                "communication_bytes": overflow_control_bytes,
                "qwd_s": 0.0,
                "refresh_s": 0.0,
                "decision": "overflow_before_gradient_communication",
                "skipped": True,
                "update_communication_s": overflow_control_s,
            }
        else:
            flat_gradient = torch.cat(
                [gradient.detach().reshape(-1) for gradient in gradients],
                out=self._flat_gradient,
            )
            feedback_before = _clone_plan_feedback(self.gradient_plan)

            def reduce_gradient() -> object:
                reduced = self.gradient_plan.execute(flat_gradient).wait()
                reduced_shard = reduced.value.float()
                norm_sq = reduced_shard[: self.layout.valid_numel].square().sum()
                torch.distributed.all_reduce(
                    norm_sq,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self.process_group,
                )
                clip = norm_clip_coefficient(norm_sq, self.grad_clip)
                reduced_shard.mul_(clip)
                return reduced_shard

            try:
                # Gradient Work commits its feedback before candidate staging.
                # Any failure until parameter publication must restore that
                # feedback, not just failures in the final qWD transaction.
                reduced_shard, gradient_communication_s = _cuda_timed(reduce_gradient)
                candidate, optimizer_s = _cuda_timed(
                    lambda: self._adamw_candidate(reduced_shard)
                )
                mode = self.schedule.mode(
                    self.step_count,
                    force_refresh=self.force_refresh,
                )
                self._copy_model_to_padded_flat()

                def commit_parameter_update() -> None:
                    work = self.qwd_plan.execute(
                        candidate.master,
                        self.model_copy_flat,
                        mode,
                    )
                    transaction = RSAGQWDTransaction(
                        publish_optimizer=self._publish_optimizer,
                        publish_model=self._publish_model,
                    )
                    transaction.commit(candidate, work)

                _, parameter_s = _cuda_timed(commit_parameter_update)
            except Exception:
                restore = getattr(
                    self.gradient_plan,
                    "_restore_committed_residual",
                    None,
                )
                if callable(restore):
                    restore(feedback_before)
                raise
            self.force_refresh = False
            self.optimizer.zero_grad(set_to_none=True)
            communication_s = (
                overflow_control_s + gradient_communication_s + parameter_s
            )
            result = {
                "update_s": optimizer_s,
                "communication_s": communication_s,
                "communication_bytes": (
                    overflow_control_bytes + self._communication_bytes(mode)
                ),
                "qwd_s": parameter_s if mode == "qwd" else 0.0,
                "refresh_s": parameter_s if mode == "fp_refresh" else 0.0,
                "decision": mode,
                "skipped": False,
                "update_communication_s": communication_s,
            }
        self.amp_scale.value = _advance_amp_scaler(scaler, overflow=overflow)
        return result

    def _flat_model(self) -> object:
        torch = _torch()
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self.parameters]
        ).contiguous()

    def _copy_model_to_padded_flat(self) -> None:
        offset = 0
        with _torch().no_grad():
            for parameter in self.parameters:
                stop = offset + parameter.numel()
                self.model_copy_flat[offset:stop].copy_(parameter.detach().reshape(-1))
                offset = stop
            self.model_copy_flat[self.global_numel :].zero_()

    def _weight_decay_shard(self) -> object:
        torch = _torch()
        values: list[object] = []
        decay_by_id = {
            id(parameter): float(group["weight_decay"])
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in self.parameters:
            values.append(
                torch.full(
                    (parameter.numel(),),
                    decay_by_id[id(parameter)],
                    device=parameter.device,
                    dtype=torch.float32,
                )
            )
        full = torch.cat(values)
        shard = torch.zeros_like(self.master)
        if self.layout.valid_numel:
            shard[: self.layout.valid_numel].copy_(
                full.narrow(
                    0,
                    self.layout.start,
                    self.layout.valid_numel,
                )
            )
        return shard

    def _adamw_candidate(self, reduced_shard: object) -> ShardedAdamW:
        candidate = self.candidate_optimizer
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        candidate._step_from_prevalidated(
            self.sharded_optimizer, reduced_shard,
            learning_rate=learning_rate, weight_decay=self.weight_decay,
        )
        candidate.learning_rate = max(learning_rate, self.base_learning_rate)
        return candidate

    def _publish_optimizer(self, candidate: object) -> None:
        if type(candidate) is not ShardedAdamW:
            raise ValueError("optimizer candidate must be exact ShardedAdamW")
        previous = self.sharded_optimizer
        self.sharded_optimizer = candidate
        self.candidate_optimizer = previous

    def _publish_model(self, flat: object) -> None:
        torch = _torch()
        if (
            type(flat) is not torch.Tensor
            or flat.dtype is not torch.float16
            or flat.device != self.parameters[0].device
            or flat.ndim != 1
            or flat.numel() != self.padded_model_numel
            or not flat.is_contiguous()
        ):
            raise ValueError("qWD committed model layout is inconsistent")
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters:
                stop = offset + parameter.numel()
                parameter.copy_(flat[offset:stop].view_as(parameter))
                offset = stop

    def _communication_bytes(self, mode: str) -> int:
        gradient = self._gradient_communication_bytes()
        if mode == "qwd":
            parameter = self.qwd_gathered_payload_bytes
        else:
            parameter = self.fp32_gathered_bytes
        return gradient + parameter

    def _gradient_communication_bytes(self) -> int:
        """Return the route's gradient byte estimate for legacy telemetry.

        Compressed layouts expose explicit payload fields. Native
        ReducedShard layouts intentionally expose zero payload metadata, so
        derive their reduce-scatter traffic from the padded shard size and
        transport dtype. Compressed payload includes self-destination slots;
        native traffic excludes self. These are different accounting scopes,
        so the legacy field is not a cross-route wire-byte measurement.
        """
        layout = self.gradient_plan.layout
        if self.gradient_route == "reduced_shard_int8_group64_ef":
            return int(layout.send_payload_bytes) + int(
                layout.receive_payload_bytes
            )
        if self.gradient_route == "reduced_shard_native_fp32":
            # execute_native pads to logical_shard_length * world_size before
            # NCCL reduce-scatter. Count both directions, excluding self.
            return (
                self.layout.padded_numel
                * self._flat_gradient.element_size()
                * 2
                * (self.world_size - 1)
            )
        raise RuntimeError("RSAG/qWD gradient route is invalid")

    def state_dict(self) -> dict[str, object]:
        return {
            "layout": {
                "global_numel": self.layout.global_numel,
                "world_size": self.layout.world_size,
                "rank": self.layout.rank,
                "start": self.layout.start,
                "valid_numel": self.layout.valid_numel,
                "padded_numel": self.layout.padded_numel,
            },
            "optimizer": self.sharded_optimizer.state_dict(),
            "gradient_feedback": {
                "key": (
                    "rsag",
                    self.rank,
                    self.world_size,
                    self.global_numel,
                    self.master.device.type,
                    self.master.device.index,
                ),
                "layout": _plan_layout_facts(self.gradient_plan),
                "residual": _clone_plan_feedback(self.gradient_plan),
            },
            "learning_rates": tuple(
                float(group["lr"]) for group in self.optimizer.param_groups
            ),
            "force_refresh": self.force_refresh,
            "gradient_route": self.gradient_route,
            "parameter_route": self.parameter_route,
        }

    def _audit_state(self) -> dict[str, object]:
        """Borrow sharded optimizer and EF tensors for synchronous hashing."""
        residual = (
            self.gradient_plan._committed_residual
            if hasattr(self.gradient_plan, "_committed_residual")
            else None
        )
        return {
            "layout": {
                "global_numel": self.layout.global_numel,
                "world_size": self.layout.world_size,
                "rank": self.layout.rank,
                "start": self.layout.start,
                "valid_numel": self.layout.valid_numel,
                "padded_numel": self.layout.padded_numel,
            },
            "optimizer": self.sharded_optimizer._audit_state(),
            "gradient_feedback": {
                "key": (
                    "rsag",
                    self.rank,
                    self.world_size,
                    self.global_numel,
                    self.master.device.type,
                    self.master.device.index,
                ),
                "layout": _plan_layout_facts(self.gradient_plan),
                "residual": residual,
            },
            "learning_rates": tuple(
                float(group["lr"]) for group in self.optimizer.param_groups
            ),
            "force_refresh": self.force_refresh,
            "gradient_route": self.gradient_route,
            "parameter_route": self.parameter_route,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fields = {
            "layout",
            "optimizer",
            "gradient_feedback",
            "learning_rates",
            "force_refresh",
        }
        if type(state) is not dict or not fields.issubset(state):
            raise ValueError("RSAG/qWD state fields are invalid")
        if set(state) - fields - {"gradient_route", "parameter_route"}:
            raise ValueError("RSAG/qWD state fields are invalid")
        checkpoint_gradient_route = state.get(
            "gradient_route", "reduced_shard_int8_group64_ef"
        )
        if (
            type(checkpoint_gradient_route) is not str
            or checkpoint_gradient_route != self.gradient_route
        ):
            raise ValueError("RSAG/qWD gradient route drifted")
        checkpoint_parameter_route = state.get(
            "parameter_route", "qwd_group64_refresh100"
        )
        if (
            type(checkpoint_parameter_route) is not str
            or checkpoint_parameter_route != self.parameter_route
        ):
            raise ValueError("RSAG/qWD parameter route drifted")
        if type(state["force_refresh"]) is not bool:
            raise ValueError("RSAG/qWD force-refresh state is invalid")
        expected_layout = {
            "global_numel": self.layout.global_numel,
            "world_size": self.layout.world_size,
            "rank": self.layout.rank,
            "start": self.layout.start,
            "valid_numel": self.layout.valid_numel,
            "padded_numel": self.layout.padded_numel,
        }
        if state["layout"] != expected_layout:
            raise ValueError("RSAG/qWD checkpoint layout is inconsistent")
        self.sharded_optimizer.load_state_dict(state["optimizer"])
        self.candidate_optimizer.master.copy_(self.master)
        self.candidate_optimizer.exp_avg.copy_(self.exp_avg)
        self.candidate_optimizer.exp_avg_sq.copy_(self.exp_avg_sq)
        self.candidate_optimizer.step_count = self.step_count
        expected_key = (
            "rsag",
            self.rank,
            self.world_size,
            self.global_numel,
            self.master.device.type,
            self.master.device.index,
        )
        _restore_plan_feedback(
            self.gradient_plan,
            state["gradient_feedback"],
            expected_key=expected_key,
            expected_device=self.master.device,
        )
        learning_rates = state["learning_rates"]
        if type(learning_rates) is not tuple or len(learning_rates) != len(
            self.optimizer.param_groups
        ):
            raise ValueError("RSAG/qWD checkpoint learning rates are inconsistent")
        for group, learning_rate in zip(
            self.optimizer.param_groups,
            learning_rates,
            strict=True,
        ):
            if type(learning_rate) is not float or learning_rate < 0.0:
                raise ValueError("RSAG/qWD checkpoint learning rate is invalid")
            group["lr"] = learning_rate
        self.force_refresh = state["force_refresh"]


def _build_cuda_plan(
    *,
    output: str,
    numel: int,
    rank: int,
    world_size: int,
    process_group: object,
) -> object:
    lowbit = import_module("lowbit_comm")
    backend_module = import_module("lowbit_comm.backends.cuda.backend")
    output_semantics = {
        "fulltensor": lowbit.OutputSemantics.FULL_TENSOR,
        "reduced_shard": lowbit.OutputSemantics.REDUCED_SHARD,
    }[output]
    collective = {
        "fulltensor": lowbit.CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        "reduced_shard": lowbit.CollectiveKind.COMPRESSED_REDUCE_SCATTER,
    }[output]
    intent = lowbit.CommunicationIntent(
        tensor=lowbit.TensorSpec(dtype="fp16", shape=(numel,)),
        shape_family=lowbit.ShapeFamily(max_numel=numel, alignment=1),
        reduction=lowbit.ReductionOp.MEAN,
        output=output_semantics,
        completion=lowbit.CompletionMode.ASYNC,
        world_size=world_size,
        rank=rank,
    )
    strategy = lowbit.StrategySpec(
        compression=lowbit.CompressionKind.INT8,
        collective=collective,
        topology=lowbit.TopologyKind.BACKEND_DEFAULT,
        group_size=64,
        error_feedback=True,
    )
    return backend_module.CudaBackend(process_group).lower(intent, strategy)


def _plan_layout_facts(plan: object) -> tuple[tuple[str, object], ...]:
    layout = plan.layout
    return tuple((item.name, getattr(layout, item.name)) for item in fields(layout))


def _clone_plan_feedback(plan: object) -> object | None:
    residual = getattr(plan, "_committed_residual", None)
    return None if residual is None else residual.detach().clone()


def _borrow_plan_feedback(plan: object) -> object | None:
    """Read a plan residual without allocating during the audit hot path."""
    return getattr(plan, "_committed_residual", None)


def _restore_plan_feedback(
    plan: object,
    state: object,
    *,
    expected_key: tuple[object, ...],
    expected_device: object,
) -> None:
    torch = _torch()
    _require_fields(state, {"key", "layout", "residual"}, "gradient feedback")
    if state["key"] != expected_key:
        raise ValueError("gradient feedback identity is inconsistent")
    if state["layout"] != _plan_layout_facts(plan):
        raise ValueError("gradient feedback layout is inconsistent")
    residual = state["residual"]
    if residual is not None:
        expected_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[plan.intent.tensor.dtype]
        if (
            type(residual) is not torch.Tensor
            or residual.device != expected_device
            or residual.dtype is not expected_dtype
            or residual.ndim != 1
            or residual.numel() != plan.intent.tensor.numel
            or not residual.is_contiguous()
        ):
            raise ValueError("gradient feedback residual is inconsistent")
    restore = getattr(plan, "_restore_committed_residual", None)
    if residual is not None or callable(restore):
        if not callable(restore):
            raise ValueError("gradient feedback plan cannot restore residual")
        restore(residual)


def _configure_ddp_communication(model, route, amp_scale, native_ddp_mode):
    if route == "native" and native_ddp_mode == "standard":
        # Preserve asynchronous DDP overlap; no per-bucket timing hook.
        return _HookTelemetry()
    return _register_ddp_hook(model, route, amp_scale)


def _register_ddp_hook(
    model: object,
    route: str,
    amp_scale: _AmpScaleState,
) -> _HookTelemetry:
    torch = _torch()
    telemetry = _HookTelemetry()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    unwrapped = model.module if hasattr(model, "module") else model
    parameter_layout_by_identity = {
        id(parameter): (
            name,
            tuple(int(value) for value in parameter.shape),
            int(parameter.numel()),
            str(parameter.dtype),
        )
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }

    def hook(_: object, bucket: object) -> object:
        buffer = bucket.buffer()
        if route == "native":

            def communicate() -> None:
                # Predivide like the standard reducer: a finite mean must not
                # overflow merely because FP16 SUM ran before averaging.
                buffer.div_(world_size)
                torch.distributed.all_reduce(buffer)

            _, elapsed_s = _cuda_timed(communicate)
            byte_count = buffer.numel() * buffer.element_size() * 2
            byte_count *= world_size - 1
        else:
            try:
                bucket_parameter_layout = tuple(
                    parameter_layout_by_identity[id(parameter)]
                    for parameter in bucket.parameters()
                )
            except KeyError as error:
                raise RuntimeError(
                    "CAG bucket contains an unknown trainable parameter"
                ) from error
            bucket_key = (
                int(bucket.index()),
                bucket_parameter_layout,
                int(buffer.numel()),
                str(buffer.dtype),
                buffer.device.type,
                buffer.device.index,
            )
            plan = telemetry.plans.get(bucket_key)
            if plan is None:
                plan = _build_cuda_plan(
                    output="fulltensor",
                    numel=buffer.numel(),
                    rank=rank,
                    world_size=world_size,
                    process_group=torch.distributed.group.WORLD,
                )
                telemetry.bind_plan(bucket_key, plan)
            buffer.mul_(1.0 / amp_scale.value)
            bucket_found_inf = torch.logical_not(torch.isfinite(buffer).all()).float()

            def reduce_bucket_overflow() -> None:
                torch.distributed.all_reduce(
                    bucket_found_inf,
                    op=torch.distributed.ReduceOp.MAX,
                    group=torch.distributed.group.WORLD,
                )

            _, overflow_control_s = _cuda_timed(reduce_bucket_overflow)
            overflow_control_bytes = bucket_found_inf.element_size() * 2
            overflow_control_bytes *= world_size - 1
            if bool(bucket_found_inf.item()):
                telemetry.add(overflow_control_s, overflow_control_bytes)
                future = torch.futures.Future()
                future.set_result(buffer)
                return future
            compressed = buffer.to(dtype=torch.float16).contiguous()
            telemetry.snapshot_feedback(bucket_key)

            def communicate() -> None:
                buffer.copy_(plan.execute(compressed).wait().value)

            _, elapsed_s = _cuda_timed(communicate)
            byte_count = int(plan.layout.gathered_payload_bytes)
            elapsed_s += overflow_control_s
            byte_count += overflow_control_bytes
        telemetry.add(elapsed_s, byte_count)
        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    hook.__annotations__["bucket"] = torch.distributed.GradBucket
    hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]
    model.register_comm_hook(state=None, hook=hook)
    return telemetry


def _torch() -> object:
    return import_module("torch")


def _unscale_and_detect_overflow(
    parameters: tuple[object, ...],
    scale: float,
    process_group: object,
    *,
    already_unscaled: bool = False,
) -> tuple[bool, tuple[object, ...], float, int]:
    torch = _torch()
    gradients = tuple(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )
    if not gradients:
        raise RuntimeError("AMP unscale requires gradients")
    device = gradients[0].device
    if any(gradient.device != device for gradient in gradients):
        raise RuntimeError("AMP gradients must share one device")
    found_inf = torch.zeros(1, device=device, dtype=torch.float32)
    inverse_scale = torch.full(
        (1,),
        1.0 if already_unscaled else 1.0 / scale,
        device=device,
        dtype=torch.float32,
    )
    torch._amp_foreach_non_finite_check_and_unscale_(
        list(gradients),
        found_inf,
        inverse_scale,
    )

    def reduce_found_inf() -> None:
        torch.distributed.all_reduce(
            found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=process_group,
        )

    _, communication_s = _cuda_timed(reduce_found_inf)
    world_size = torch.distributed.get_world_size(process_group)
    communication_bytes = found_inf.numel() * found_inf.element_size() * 2
    communication_bytes *= world_size - 1
    return (
        bool(found_inf.item()),
        gradients,
        communication_s,
        communication_bytes,
    )


def _advance_amp_scaler(scaler: object, *, overflow: bool) -> float:
    state = scaler.state_dict()
    scale = float(state["scale"])
    if overflow:
        state["scale"] = max(1.0, scale * float(state["backoff_factor"]))
        state["_growth_tracker"] = 0
    else:
        tracker = int(state["_growth_tracker"]) + 1
        if tracker >= int(state["growth_interval"]):
            state["scale"] = scale * float(state["growth_factor"])
            tracker = 0
        state["_growth_tracker"] = tracker
    scaler.load_state_dict(state)
    return float(state["scale"])


def _require_fields(
    value: object,
    fields: set[str],
    name: str,
) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")


def _import_psi_source(root: str | Path) -> object:
    source = Path(root).resolve()
    if not (source / "psi_policy" / "train.py").is_file():
        raise ValueError("PSI source does not contain psi_policy/train.py")
    _validate_psi_source_contract(source)
    sys.dont_write_bytecode = True
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import_module("psi_policy.train")
    return import_module("psi_policy.workspace.train_workspace")


def _validate_psi_source_contract(source: Path) -> None:
    violations = [
        relative
        for relative in _DEPRECATED_PSI_COMMUNICATION_FILES
        if (source / relative).is_file()
    ]
    workspace = source / "psi_policy" / "workspace" / "train_workspace.py"
    if not workspace.is_file():
        raise ValueError(
            "PSI source does not contain psi_policy/workspace/train_workspace.py"
        )
    runtime_sources = [workspace]
    communication_init = source / "psi_policy" / "communication" / "__init__.py"
    if communication_init.is_file():
        runtime_sources.append(communication_init)
    violations.extend(
        f"{path.relative_to(source).as_posix()}:{marker}"
        for path in runtime_sources
        for marker in _DEPRECATED_PSI_COMMUNICATION_IMPORTS
        if marker in path.read_text(encoding="utf-8")
    )
    if violations:
        rendered = ", ".join(sorted(set(violations)))
        raise ValueError(
            f"deprecated PSI communication integration is not supported: {rendered}"
        )


def _build_workspace(args: object, rank: int, world_size: int) -> tuple[object, ...]:
    source = Path(args.psi_source).resolve()
    workspace_module = _import_psi_source(source)
    hydra = import_module("hydra")
    _reject_locked_psi_overrides(args.psi_override)
    overrides = [
        f"training.seed={args.seed}",
        f"training.num_epochs={args.epochs}",
        f"train_dataloader.batch_size={args.batch_size}",
        "train_dataloader.num_workers=0",
        "train_dataloader.persistent_workers=false",
        "val_dataloader.num_workers=0",
        "val_dataloader.persistent_workers=false",
        "training.gradient_accumulate_every=1",
        "training.use_ema=false",
        "notifications.feishu.enabled=false",
        "logging.mode=disabled",
        *args.psi_override,
    ]
    config_dir = source / "psi_policy" / "config"
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        config = hydra.compose(
            config_name="train_workspace",
            overrides=overrides,
        )
    workspace = workspace_module.TrainWorkspace(config)
    torch = _torch()
    accelerator = SimpleNamespace(
        is_main_process=rank == 0,
        num_processes=world_size,
        device=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    workspace.prepare(accelerator=accelerator)
    train_loader, train_sampler = workspace._build_train_dataloader()
    val_loader, val_sampler = workspace._build_val_dataloader()
    # TrainWorkspace.prepare() loads normalizers; it does not apply Accelerate's
    # batch sharding. PSI's composed sampler emits every rank's batches.
    train_sampler = RankBatchSampler(
        train_sampler,
        rank=rank,
        world_size=world_size,
        batch_size=train_loader.batch_size,
    )
    val_sampler = RankBatchSampler(
        val_sampler, rank=rank, world_size=world_size, batch_size=val_loader.batch_size
    )
    train_loader = _configure_data_loader(train_loader, args, rank=rank, stream=0)
    val_loader = _configure_data_loader(val_loader, args, rank=rank, stream=1)
    train_loader = _resume_loader(train_loader, train_sampler)
    val_loader = _resume_loader(val_loader, val_sampler)
    return workspace, train_loader, train_sampler, val_loader, val_sampler


def _reject_locked_psi_overrides(overrides: object) -> None:
    if type(overrides) is not list or not all(
        type(value) is str for value in overrides
    ):
        raise ValueError("PSI overrides must be an exact string list")
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if any(
            key == locked
            or key.startswith(locked + ".")
            or locked.startswith(key + ".")
            for locked in _LOCKED_OVERRIDE_KEYS
        ):
            raise ValueError(f"PSI override changes locked parity setting: {key}")


def _parameter_sha256(model: object) -> str:
    digest = sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _convert_model_to_common_fp16(model: object) -> None:
    _convert_model_precision(model, "fp16")


def _convert_model_precision(model: object, precision: str) -> None:
    if precision not in {"fp16", "fp32"}:
        raise ValueError("invalid model precision")
    torch = _torch()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=dtype)


def _object_sha256(value: object) -> str:
    torch = _torch()
    buffer = BytesIO()
    torch.save(value, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def _state_sha256(value: object) -> str:
    "Hash nested optimizer state independently of pickle storage identities."
    torch = _torch()
    digest = sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                update(key)
                update(item[key])
            return
        if type(item) in {list, tuple}:
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            for child in item:
                update(child)
            return
        if item is None or type(item) in {bool, int, float, str}:
            digest.update(type(item).__name__.encode("ascii") + b"\0")
            digest.update(repr(item).encode("utf-8") + b"\0")
            return
        raise TypeError(f"unsupported state value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _rng_sha256() -> str:
    torch = _torch()
    state: dict[str, object] = {
        "python": import_module("random").getstate(),
        "numpy": import_module("numpy").random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return _object_sha256(state)


def _capture_rng_state() -> dict[str, object]:
    torch = _torch()
    return {
        "python": import_module("random").getstate(),
        "numpy": import_module("numpy").random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    torch = _torch()
    import_module("random").setstate(state["python"])
    import_module("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _materialize_sampler_indices(
    sampler: object, loader_length: int
) -> tuple[int, ...]:
    if sampler is None:
        return tuple(range(loader_length))
    return tuple(int(value) for value in sampler)


def _configure_data_loader(loader: object, args: object, *, rank: int, stream: int) -> object:
    config = loader_configuration(
        getattr(args, "data_mode", "legacy"), getattr(args, "loader_workers", 0),
        getattr(args, "loader_prefetch_factor", 2), args.seed,
    )
    if config["mode"] != "legacy":
        loader._ccdl_source_loader = loader
        loader._ccdl_prefetch_options = {
            "base_seed": data_loader_seed(config["seed"], rank, stream),
            "num_workers": config["workers"],
            "prefetch_factor": config["prefetch_factor"],
        }
    return loader


def _resume_loader(loader: object, indices: object, *, epoch: int = 0,
                   start_position: int = 0) -> object:
    if hasattr(loader, "_ccdl_prefetch_options"):
        from tests.benchmarks.psi_prefetch import build_prefetch_loader

        source = loader._ccdl_source_loader
        options = loader._ccdl_prefetch_options
        result = build_prefetch_loader(
            source, tuple(indices), epoch=epoch, start_position=start_position, **options
        )
        result._ccdl_source_loader = source
        result._ccdl_prefetch_options = options
        return result
    torch = _torch()
    if (
        int(loader.num_workers) != 0
        or bool(loader.persistent_workers)
        or loader.batch_size is None
    ):
        raise RuntimeError("formal resume requires a synchronous batched loader")
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=int(loader.batch_size),
        sampler=tuple(indices)[start_position:] if start_position else indices,
        num_workers=0,
        collate_fn=loader.collate_fn,
        pin_memory=bool(loader.pin_memory),
        drop_last=bool(loader.drop_last),
        timeout=0,
        worker_init_fn=None,
        generator=loader.generator,
        persistent_workers=False,
    )


def _iterator_preserving_rng(loader: object) -> object:
    torch = _torch()
    random = import_module("random")
    numpy = import_module("numpy")
    state = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "loader": _loader_rng_state(loader),
    }
    iterator = iter(loader)
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    _restore_loader_rng_state(loader, state["loader"])
    return iterator


def _to_device(value: object, device: object) -> object:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if type(value) is dict:
        return {key: _to_device(item, device) for key, item in value.items()}
    if type(value) is list:
        return [_to_device(item, device) for item in value]
    if type(value) is tuple:
        return tuple(_to_device(item, device) for item in value)
    return value


def _reporting_augmentation_sha256(
    *,
    workspace: object,
    batch: object,
    device: object,
    augmentation_rng: object,
    current_rng: object,
) -> str:
    augmented_batch: object | None = None
    try:
        _restore_rng_state(augmentation_rng)
        augmented_batch = workspace._apply_train_augmentation(_to_device(batch, device))
        return _state_sha256(augmented_batch)
    finally:
        del augmented_batch
        _restore_rng_state(current_rng)


def _rank_gap(model: object, process_group: object) -> float:
    torch = _torch()
    flat = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )
    reference = flat.clone()
    torch.distributed.broadcast(reference, src=0, group=process_group)
    gap = (flat - reference).abs().max().float()
    torch.distributed.all_reduce(
        gap,
        op=torch.distributed.ReduceOp.MAX,
        group=process_group,
    )
    return float(gap)


def _gpu_telemetry(
    physical_gpu_ids: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ()
    rows_by_index: dict[int, list[str]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            reported_index = int(fields[0])
        except ValueError:
            continue
        if reported_index in rows_by_index:
            return ()
        rows_by_index[reported_index] = fields
    facts = []
    for physical_gpu_id in physical_gpu_ids:
        fields = rows_by_index.get(physical_gpu_id)
        if fields is None:
            return ()
        facts.append(
            {
                "gpu": physical_gpu_id,
                "utilization": float(fields[1]),
                "memory_used_mib": float(fields[2]),
                "temperature_c": float(fields[3]),
                "sm_clock_mhz": float(fields[4]),
            }
        )
    return tuple(facts)


def _rank_device_identity(
    *,
    global_rank: int,
    local_rank: int,
    node_rank: int,
) -> dict[str, object]:
    """Return fail-closed physical GPU identity for one global rank."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    tokens = tuple(value.strip() for value in visible.split(",") if value.strip())
    visible_device = tokens[local_rank] if local_rank < len(tokens) else ""
    if not visible_device:
        raise RuntimeError("rank device identity has no visible device token")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("rank device identity query failed") from error
    inventory: list[tuple[int, str, str]] = []
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        inventory.append((index, fields[1], fields[2]))
    matches = [
        item
        for item in inventory
        if visible_device in {str(item[0]), item[1]}
        or item[1].startswith(visible_device)
    ]
    if len(matches) != 1:
        raise RuntimeError("rank device identity did not match one GPU")
    physical_index, gpu_uuid, pci_bus_id = matches[0]
    return {
        "hostname": socket.gethostname(),
        "node_rank": node_rank,
        "global_rank": global_rank,
        "local_rank": local_rank,
        "visible_device": visible_device,
        "physical_gpu_index": physical_index,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
    }


def _gather_rank_device_identities(
    torch: object,
    *,
    global_rank: int,
    local_rank: int,
    node_rank: int,
    world_size: int,
    process_group: object,
) -> tuple[dict[str, object], ...]:
    """Gather rank-local physical identities without synthesized fallbacks."""
    local = _rank_device_identity(
        global_rank=global_rank,
        local_rank=local_rank,
        node_rank=node_rank,
    )
    gathered: list[object] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        gathered,
        local,
        group=process_group,
    )
    if not all(type(item) is dict for item in gathered):
        raise RuntimeError("rank device identity gather returned invalid rows")
    return tuple(dict(item) for item in gathered)


def _summarize_gpu_samples(
    identity: object,
    samples: object,
    *,
    window_s: float,
) -> dict[str, object]:
    """Summarize rank-local samples without inventing missing telemetry."""
    if type(identity) is not dict:
        raise ValueError("GPU telemetry identity must be an exact dict")
    if type(samples) is not tuple or not samples:
        raise ValueError("GPU telemetry requires at least one sample")
    if type(window_s) is not float or not isfinite(window_s) or window_s < 0.0:
        raise ValueError("GPU telemetry window must be non-negative")
    columns: list[list[float]] = [[], [], [], []]
    for sample in samples:
        if type(sample) is not tuple or len(sample) != 4:
            raise ValueError("GPU telemetry sample fields are invalid")
        if not all(
            type(value) is float and isfinite(value) and value >= 0.0
            for value in sample
        ):
            raise ValueError("GPU telemetry sample values are invalid")
        for column, value in zip(columns, sample, strict=True):
            column.append(value)
    for column in columns:
        column.sort()

    def stats(values: list[float], *, include_p50: bool) -> dict[str, float]:
        prefix = ""
        result = {
            f"{prefix}mean": sum(values) / len(values),
            f"{prefix}p95": _percentile(values, 0.95),
            f"{prefix}max": values[-1],
        }
        if include_p50:
            result[f"{prefix}p50"] = _percentile(values, 0.50)
        return result

    utilization = stats(columns[0], include_p50=True)
    memory = stats(columns[1], include_p50=True)
    temperature = stats(columns[2], include_p50=False)
    clock = stats(columns[3], include_p50=False)
    return {
        "global_rank": identity["global_rank"],
        "hostname": identity["hostname"],
        "gpu_uuid": identity["gpu_uuid"],
        "pci_bus_id": identity["pci_bus_id"],
        "physical_gpu_index": identity["physical_gpu_index"],
        "sample_count": len(samples),
        "window_s": window_s,
        "utilization_mean": utilization["mean"],
        "utilization_p50": utilization["p50"],
        "utilization_p95": utilization["p95"],
        "utilization_max": utilization["max"],
        "memory_used_mib_mean": memory["mean"],
        "memory_used_mib_p50": memory["p50"],
        "memory_used_mib_p95": memory["p95"],
        "memory_used_mib_max": memory["max"],
        "temperature_c_mean": temperature["mean"],
        "temperature_c_p95": temperature["p95"],
        "temperature_c_max": temperature["max"],
        "sm_clock_mhz_mean": clock["mean"],
        "sm_clock_mhz_p95": clock["p95"],
        "sm_clock_mhz_max": clock["max"],
    }


class _GpuTelemetrySampler:
    """Sample one rank's GPU throughout the measured process window."""

    def __init__(self, identity: dict[str, object], interval_s: float = 1.0):
        if type(interval_s) is not float or interval_s <= 0.0:
            raise ValueError("GPU telemetry interval must be positive")
        self._identity = dict(identity)
        self._interval_s = interval_s
        self._samples: list[tuple[float, float, float, float]] = []
        self._stop = Event()
        self._thread: Thread | None = None
        self._started = 0.0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU telemetry sampler is already started")
        self._started = time.perf_counter()
        self._thread = Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self.close()
        return _summarize_gpu_samples(
            self._identity,
            tuple(self._samples),
            window_s=float(time.perf_counter() - self._started),
        )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self._interval_s * 2.0))
            if self._thread.is_alive():
                raise RuntimeError("GPU telemetry sampler did not stop")
            self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            sample = self._query_sample()
            if sample is not None:
                self._samples.append(sample)
            self._stop.wait(self._interval_s)

    def _query_sample(self) -> tuple[float, float, float, float] | None:
        command = [
            "nvidia-smi",
            f"--id={self._identity['gpu_uuid']}",
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            ).stdout.strip()
            fields = tuple(field.strip() for field in output.split(","))
            if len(fields) != 4:
                return None
            values = tuple(float(field) for field in fields)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if not all(isfinite(value) and value >= 0.0 for value in values):
            return None
        return values


def _amp_configuration_dict(
    *,
    initial_scale: float,
    effective_start_scale: float,
    scaler_state: dict[str, object],
) -> dict[str, object]:
    return {
        "precision": "fp16",
        "enabled": True,
        "initial_scale": float(initial_scale),
        "effective_start_scale": float(effective_start_scale),
        "growth_interval": int(scaler_state["growth_interval"]),
        "growth_factor": float(scaler_state["growth_factor"]),
        "backoff_factor": float(scaler_state["backoff_factor"]),
    }


def _amp_configuration_tuple(
    value: dict[str, object],
) -> tuple[str, bool, float, float, int, float, float]:
    return (
        str(value["precision"]),
        bool(value["enabled"]),
        float(value["initial_scale"]),
        float(value["effective_start_scale"]),
        int(value["growth_interval"]),
        float(value["growth_factor"]),
        float(value["backoff_factor"]),
    )


def _validate_amp_configuration(
    value: object,
    *,
    scaler_state: object,
) -> None:
    fields = {
        "precision",
        "enabled",
        "initial_scale",
        "effective_start_scale",
        "growth_interval",
        "growth_factor",
        "backoff_factor",
    }
    _require_fields(value, fields, "checkpoint AMP configuration")
    _require_fields(
        scaler_state,
        {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        },
        "checkpoint scaler",
    )
    if (
        value["precision"] != "fp16"
        or value["enabled"] is not True
        or type(value["initial_scale"]) is not float
        or not isfinite(value["initial_scale"])
        or value["initial_scale"] <= 0.0
        or type(value["effective_start_scale"]) is not float
        or not isfinite(value["effective_start_scale"])
        or value["effective_start_scale"] <= 0.0
        or type(value["growth_interval"]) is not int
        or value["growth_interval"] <= 0
        or type(value["growth_factor"]) is not float
        or not isfinite(value["growth_factor"])
        or value["growth_factor"] <= 1.0
        or type(value["backoff_factor"]) is not float
        or not isfinite(value["backoff_factor"])
        or not 0.0 < value["backoff_factor"] < 1.0
        or value["effective_start_scale"] != float(scaler_state["scale"])
        or value["growth_interval"] != int(scaler_state["growth_interval"])
        or value["growth_factor"] != float(scaler_state["growth_factor"])
        or value["backoff_factor"] != float(scaler_state["backoff_factor"])
    ):
        raise ValueError("checkpoint AMP configuration is inconsistent")


def _clone_checkpoint_value(value: object) -> object:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if type(value) is dict:
        return {key: _clone_checkpoint_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_checkpoint_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_checkpoint_value(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _validate_checkpoint_value(value: object) -> None:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.device.type != "cuda" and value.device.type != "cpu":
            raise ValueError("checkpoint batch tensor device is invalid")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("checkpoint batch keys must be exact strings")
            _validate_checkpoint_value(item)
        return
    if type(value) in {list, tuple}:
        for item in value:
            _validate_checkpoint_value(item)
        return
    if value is None or type(value) in {bool, int, float, str}:
        return
    raise ValueError("checkpoint batch value is invalid")


def _loader_generators(loader: object) -> tuple[tuple[str, object], ...]:
    candidates = (
        ("loader", getattr(loader, "generator", None)),
        (
            "sampler",
            getattr(getattr(loader, "sampler", None), "generator", None),
        ),
        (
            "batch_sampler",
            getattr(
                getattr(getattr(loader, "batch_sampler", None), "sampler", None),
                "generator",
                None,
            ),
        ),
    )
    result = []
    seen: set[int] = set()
    for name, generator in candidates:
        if generator is None or id(generator) in seen:
            continue
        if not isinstance(generator, _torch().Generator):
            raise ValueError("loader generator is inconsistent")
        seen.add(id(generator))
        result.append((name, generator))
    return tuple(result)


def _loader_rng_state(loader: object) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name, generator.get_state().cpu().clone())
        for name, generator in _loader_generators(loader)
    )


def _validate_loader_rng_state(value: object) -> None:
    torch = _torch()
    if type(value) is not tuple:
        raise ValueError("loader RNG state must be an exact tuple")
    names: set[str] = set()
    for entry in value:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or entry[0] in names
            or type(entry[1]) is not torch.Tensor
            or entry[1].device.type not in {"cpu", "cuda"}
            or entry[1].dtype is not torch.uint8
            or entry[1].ndim != 1
        ):
            raise ValueError("loader RNG state is inconsistent")
        names.add(entry[0])


def _restore_loader_rng_state(loader: object, value: object) -> None:
    _validate_loader_rng_state(value)
    expected = dict(_loader_generators(loader))
    saved = dict(value)
    if set(expected) != set(saved):
        raise ValueError("loader RNG generators are inconsistent")
    for name, generator in expected.items():
        generator.set_state(saved[name].cpu())


def _save_checkpoint(
    path: Path,
    *,
    route: str,
    epoch: int,
    step: int,
    step_in_epoch: int,
    next_batch_indices: tuple[int, ...],
    model: object,
    engine: object,
    scheduler: object,
    scaler: object,
    amp_initial_scale: float,
    amp_scale: _AmpScaleState,
    warmup_batch: object,
    train_loader: object,
) -> None:
    torch = _torch()
    scaler_state = scaler.state_dict()
    payload = {
        "training_protocol": dict(engine.training_protocol),
        "route": route,
        "epoch": epoch,
        "step": step,
        "step_in_epoch": step_in_epoch,
        "next_batch_indices": next_batch_indices,
        "model": model.state_dict(),
        "engine": engine.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler_state,
        "amp_configuration": _amp_configuration_dict(
            initial_scale=amp_initial_scale,
            effective_start_scale=amp_scale.value,
            scaler_state=scaler_state,
        ),
        "warmup_batch": _clone_checkpoint_value(warmup_batch),
        "loader_rng": _loader_rng_state(train_loader),
        "rng": {
            "python": import_module("random").getstate(),
            "numpy": import_module("numpy").random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    route: str,
    model: object,
    engine: object,
    scheduler: object,
    scaler: object,
    amp_scale: _AmpScaleState,
    train_loader: object,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    torch = _torch()
    if payload is None:
        payload = _read_checkpoint_payload(path, route=route)
    else:
        _validate_checkpoint_payload(payload, route=route)
    if payload["training_protocol"] != engine.training_protocol:
        raise ValueError("checkpoint training protocol is inconsistent")
    model.load_state_dict(payload["model"])
    engine.load_state_dict(payload["engine"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    amp_scale.value = float(payload["amp_configuration"]["effective_start_scale"])
    random = import_module("random")
    numpy = import_module("numpy")
    random.setstate(payload["rng"]["python"])
    numpy.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
    _restore_loader_rng_state(train_loader, payload["loader_rng"])
    return payload


def _read_checkpoint_payload(
    path: Path,
    *,
    route: str,
) -> dict[str, object]:
    payload = _torch().load(path, map_location="cuda", weights_only=False)
    _validate_checkpoint_payload(payload, route=route)
    return payload


def _validate_checkpoint_payload(payload: object, *, route: str) -> None:
    fields = {
        "training_protocol",
        "route",
        "epoch",
        "step",
        "step_in_epoch",
        "next_batch_indices",
        "model",
        "engine",
        "scheduler",
        "scaler",
        "amp_configuration",
        "warmup_batch",
        "loader_rng",
        "rng",
    }
    _require_fields(payload, fields, "checkpoint")
    validate_training_protocol(payload["training_protocol"])
    if payload["route"] != route:
        raise ValueError("checkpoint route is inconsistent")
    _validate_amp_configuration(
        payload["amp_configuration"],
        scaler_state=payload["scaler"],
    )
    _validate_checkpoint_value(payload["warmup_batch"])
    _validate_loader_rng_state(payload["loader_rng"])
    if (
        type(payload["epoch"]) is not int
        or payload["epoch"] < 0
        or type(payload["step"]) is not int
        or payload["step"] < 0
        or type(payload["step_in_epoch"]) is not int
        or payload["step_in_epoch"] < 0
        or type(payload["next_batch_indices"]) is not tuple
        or not all(type(value) is int for value in payload["next_batch_indices"])
    ):
        raise ValueError("checkpoint position is inconsistent")


def _resolve_resume_path(value: str, rank: int) -> Path:
    if type(value) is not str or value.count("{rank}") != 1:
        raise ValueError("resume path must contain one exact {rank} placeholder")
    expanded = value.replace("{rank}", str(rank))
    if "{" in expanded or "}" in expanded:
        raise ValueError("resume path contains an unknown placeholder")
    return Path(expanded)


def _write_resume_oracle(path: Path, facts: ResumeFacts) -> None:
    value = {
        "next_batch_indices": list(facts.next_batch_indices),
        "next_batch_sha256": facts.next_batch_sha256,
        "next_augmentation_sha256": facts.next_augmentation_sha256,
        "learning_rate": facts.learning_rate,
        "amp_scale": facts.amp_scale,
        "optimizer_state_sha256": facts.optimizer_state_sha256,
        "model_sha256": facts.model_sha256,
        "next_loss": facts.next_loss,
        "post_learning_rate": facts.post_learning_rate,
        "post_amp_scale": facts.post_amp_scale,
        "post_optimizer_state_sha256": facts.post_optimizer_state_sha256,
        "post_model_sha256": facts.post_model_sha256,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prepare_resume_oracle_observation(
    *,
    path: Path,
    next_batch_indices: tuple[int, ...],
    learning_rate: float,
    amp_scale: float,
    optimizer_state_sha256: str,
    model_sha256: str,
) -> tuple[Path, tuple[int, ...], float, float, str, str]:
    """Freeze read-only facts for the next uninterrupted oracle observation."""
    return (
        path,
        next_batch_indices,
        learning_rate,
        amp_scale,
        optimizer_state_sha256,
        model_sha256,
    )


def prepare_epoch_resume_oracle_observation(
    *, path: Path, next_batch_indices: tuple[int, ...], learning_rate: float,
    amp_scale: float, observed: dict[str, object],
) -> tuple[Path, tuple[int, ...], float, float, str, str]:
    """Carry the last step's actual audited hashes into the next epoch oracle."""
    return _prepare_resume_oracle_observation(
        path=path,
        next_batch_indices=next_batch_indices,
        learning_rate=learning_rate,
        amp_scale=amp_scale,
        optimizer_state_sha256=observed["optimizer_sha256"],
        model_sha256=observed["model_sha256"],
    )


def _load_resume_oracle(path: Path) -> ResumeFacts:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "next_batch_indices",
        "next_batch_sha256",
        "next_augmentation_sha256",
        "learning_rate",
        "amp_scale",
        "optimizer_state_sha256",
        "model_sha256",
        "next_loss",
        "post_learning_rate",
        "post_amp_scale",
        "post_optimizer_state_sha256",
        "post_model_sha256",
    }
    _require_fields(value, fields, "resume oracle")
    return ResumeFacts(
        next_batch_indices=tuple(value["next_batch_indices"]),
        next_batch_sha256=value["next_batch_sha256"],
        next_augmentation_sha256=value["next_augmentation_sha256"],
        learning_rate=value["learning_rate"],
        amp_scale=value["amp_scale"],
        optimizer_state_sha256=value["optimizer_state_sha256"],
        model_sha256=value["model_sha256"],
        next_loss=value["next_loss"],
        post_learning_rate=value["post_learning_rate"],
        post_amp_scale=value["post_amp_scale"],
        post_optimizer_state_sha256=value["post_optimizer_state_sha256"],
        post_model_sha256=value["post_model_sha256"],
    )


def _write_raw_records(path: Path, records: list[dict[str, object]]) -> None:
    "Validate and account canonical preparation before publishing raw rows."
    lines = []
    for record in records:
        started = time.perf_counter()
        validate_step_record(record)
        json.dumps(record, sort_keys=True)
        elapsed_s = time.perf_counter() - started
        if record["timing"]["report_serialization_s"] is not None:
            record["timing"]["report_serialization_s"] += elapsed_s
        validate_step_record(record)
        lines.append(json.dumps(record, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_epoch(
    model: object,
    loader: object,
    device: object,
    seed: int,
) -> float:
    torch = _torch()
    random = import_module("random")
    numpy = import_module("numpy")
    rng = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "loader": _loader_rng_state(loader),
    }
    module_training = tuple(
        (module, bool(module.training)) for module in model.modules()
    )
    try:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for _, generator in _loader_generators(loader):
            generator.manual_seed(seed)
        loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
        sample_count = torch.zeros(1, device=device, dtype=torch.float64)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_size = _batch_sample_count(batch)
                value = _to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(value, training=True)
                loss_sum.add_(loss.detach().double() * batch_size)
                sample_count.add_(batch_size)
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sample_count, op=torch.distributed.ReduceOp.SUM)
        if sample_count.item() == 0.0:
            return 0.0
        return float((loss_sum / sample_count).item())
    finally:
        for module, training in module_training:
            module.training = training
        random.setstate(rng["python"])
        numpy.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        _restore_loader_rng_state(loader, rng["loader"])


def _batch_sample_count(value: object) -> int:
    torch = _torch()
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if type(value) is dict:
        for item in value.values():
            try:
                return _batch_sample_count(item)
            except ValueError:
                continue
    if type(value) in {list, tuple}:
        for item in value:
            try:
                return _batch_sample_count(item)
            except ValueError:
                continue
    raise ValueError("validation batch does not contain a sample tensor")


def _scheduler_trajectory(workspace: object, total_steps: int) -> tuple[float, ...]:
    torch = _torch()
    scheduler_module = import_module("psi_policy.model.common.lr_scheduler")
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=float(workspace.cfg.optimizer.lr),
    )
    scheduler = scheduler_module.get_scheduler(
        workspace.cfg.training.lr_scheduler,
        optimizer,
        num_warmup_steps=int(workspace.cfg.training.lr_warmup_steps),
        num_training_steps=total_steps,
        last_epoch=-1,
    )
    values = []
    for _ in range(total_steps):
        values.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return tuple(values)


def _stable_ddp_bucket_cap_mb(model: object) -> int:
    "Keep DDP's initial and rebuilt layouts to one trainable-parameter bucket."
    mib = 1024 * 1024
    trainable_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return max(1, trainable_bytes // mib + 1)


def _stabilize_ddp_bucket_layout(
    model: object,
    workspace: object,
    batch: object,
    device: object,
    telemetry: _HookTelemetry | None,
) -> None:
    """Run the common model warmup without changing training state."""
    torch = _torch()
    random = import_module("random")
    numpy = import_module("numpy")
    rng = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    buffers = tuple((buffer, buffer.detach().clone()) for buffer in model.buffers())
    gradients = tuple(
        (
            parameter,
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in model.parameters()
    )
    module_training = tuple(
        (module, bool(module.training)) for module in model.modules()
    )
    try:
        for _ in range(_DDP_BUCKET_WARMUP_BACKWARDS):
            device_batch = _to_device(batch, device)
            model_batch = workspace._apply_train_augmentation(device_batch)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = model(model_batch, training=True)
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, saved_grad, saved_value in gradients:
            parameter.grad = saved_grad
            if saved_grad is not None:
                saved_grad.copy_(saved_value)
        if telemetry is not None:
            telemetry.reset_after_warmup()
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, training in module_training:
            module.training = training
        random.setstate(rng["python"])
        numpy.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])


def _run(args: object) -> None:
    global _PHASE_TIMER
    if args.model_precision not in {"fp16", "fp32"}:
        raise ValueError("invalid model precision")
    if args.model_precision == "fp32" and (
        args.route != "native" or args.native_ddp_mode != "standard"
    ):
        raise ValueError("FP32 model reference requires native route with standard DDP")
    _PHASE_TIMER = PhaseTimer(args.timing_mode, lambda: _torch().cuda)
    process_started = time.perf_counter()
    torch = _torch()
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    process_group = torch.distributed.group.WORLD
    gpu_sampler: _GpuTelemetrySampler | None = None
    try:
        rank_devices = _gather_rank_device_identities(
            torch,
            global_rank=rank,
            local_rank=local_rank,
            node_rank=int(os.environ.get("GROUP_RANK", "0")),
            world_size=world_size,
            process_group=process_group,
        )
        gpu_sampler = _GpuTelemetrySampler(rank_devices[rank])
        gpu_sampler.start()
        workspace, train_loader, train_sampler, val_loader, val_sampler = (
            _build_workspace(args, rank, world_size)
        )
        if args.inject_overflow_step < 0:
            raise ValueError("inject_overflow_step must be non-negative")
        if args.amp_growth_interval <= 0:
            raise ValueError("amp_growth_interval must be positive")
        if args.resume_oracle_mode == "require" and args.resume is None:
            raise ValueError("resume oracle require mode needs --resume")
        if args.resume_oracle_mode == "write" and args.resume is not None:
            raise ValueError("resume oracle write mode requires an uninterrupted run")
        epoch_start = 0
        global_step = 0
        resume_step_in_epoch = 0
        resume_path: Path | None = None
        resume_payload: dict[str, object] | None = None
        if args.resume is not None:
            resume_path = _resolve_resume_path(args.resume, rank)
            resume_payload = _read_checkpoint_payload(
                resume_path,
                route=args.route,
            )
            epoch_start = int(resume_payload["epoch"])
            global_step = int(resume_payload["step"])
            resume_step_in_epoch = int(resume_payload["step_in_epoch"])
        sampler_indices = _materialize_sampler_indices(
            train_sampler,
            len(train_loader),
        )
        augmentation_rng_sha256 = _rng_sha256()
        total_steps = max(1, len(train_loader) * args.epochs)
        lr_schedule = _scheduler_trajectory(workspace, total_steps)
        model_parameter_count = sum(
            parameter.numel() for parameter in workspace.model.parameters()
        )
        workspace.model.to(device)
        model = workspace.model
        if args.model_precision == "fp16":
            _convert_model_to_common_fp16(model)
        else:
            _convert_model_precision(model, args.model_precision)
        ddp_bucket_cap_mb = _stable_ddp_bucket_cap_mb(model)
        if args.route == "native" and args.native_ddp_mode == "standard":
            ddp_bucket_cap_mb = 25
        scheduler_module = import_module("psi_policy.model.common.lr_scheduler")
        scheduler = scheduler_module.get_scheduler(
            workspace.cfg.training.lr_scheduler,
            workspace.optimizer,
            num_warmup_steps=int(workspace.cfg.training.lr_warmup_steps),
            num_training_steps=total_steps,
            last_epoch=-1,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=args.amp_initial_scale,
            growth_interval=args.amp_growth_interval,
        )
        configured_scaler_state = scaler.state_dict()
        amp_scale = _AmpScaleState(float(configured_scaler_state["scale"]))
        if resume_payload is not None:
            checkpoint_amp = resume_payload["amp_configuration"]
            configured_amp = _amp_configuration_dict(
                initial_scale=args.amp_initial_scale,
                effective_start_scale=amp_scale.value,
                scaler_state=configured_scaler_state,
            )
            for field in (
                "precision",
                "enabled",
                "initial_scale",
                "growth_interval",
                "growth_factor",
                "backoff_factor",
            ):
                if checkpoint_amp[field] != configured_amp[field]:
                    raise ValueError(f"checkpoint AMP configuration drifted: {field}")
        replay_batch: object | None = None
        replay_iterator: object | None = None
        replay_epoch_indices: tuple[int, ...] | None = None
        if resume_payload is None:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_start)
            replay_epoch_indices = _materialize_sampler_indices(
                train_sampler,
                len(train_loader),
            )
            replay_iterator = iter(_resume_loader(train_loader, replay_epoch_indices))
            try:
                replay_batch = next(replay_iterator)
            except StopIteration as error:
                raise RuntimeError("model warmup requires one real training batch") from error
        else:
            replay_batch = resume_payload["warmup_batch"]
        if args.route in {"native", "cag"}:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                bucket_cap_mb=ddp_bucket_cap_mb,
            )
            telemetry = _configure_ddp_communication(
                model, args.route, amp_scale, args.native_ddp_mode
            )
        else:
            telemetry = _HookTelemetry()
        _stabilize_ddp_bucket_layout(
            model,
            workspace,
            replay_batch,
            device,
            telemetry if args.route in {"native", "cag"} else None,
        )
        engine = build_engine(
            args.route,
            native_factory=lambda: NativeUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                telemetry=telemetry,
                amp_scale=amp_scale,
            ),
            cag_factory=lambda: CAGUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                telemetry=telemetry,
                amp_scale=amp_scale,
            ),
            rsag_qwd_factory=lambda: RSAGQWDUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                rank=rank,
                world_size=world_size,
                process_group=process_group,
                amp_scale=amp_scale,
                parameter_route=args.rsag_parameter_route,
                gradient_route=args.rsag_gradient_route,
            ),
        )
        unwrapped = model.module if hasattr(model, "module") else model
        engine.training_protocol = training_protocol(
            model_precision=args.model_precision,
            rank=rank,
            world_size=world_size,
            batch_size=args.batch_size,
            native_ddp_mode=args.native_ddp_mode,
            timing_mode=args.timing_mode,
            data_mode=args.data_mode,
            loader_workers=args.loader_workers,
            loader_prefetch_factor=args.loader_prefetch_factor,
            seed=args.seed,
        )
        if args.route == "native" and args.native_ddp_mode == "standard":
            engine.gradient_route = "ddp_nccl_standard_uninstrumented"
        initial_sha256 = _parameter_sha256(unwrapped)
        if args.probe_only:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "route": args.route,
                            "parity": {
                                "initial_parameter_sha256": initial_sha256,
                                "sampler_indices_sha256": canonical_sha256(
                                    sampler_indices
                                ),
                                "augmentation_rng_sha256": (augmentation_rng_sha256),
                                "lr_schedule_sha256": canonical_sha256(lr_schedule),
                                "model_parameter_count": model_parameter_count,
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return
        resume_oracle: ResumeFacts | None = None
        resume_oracle_path: Path | None = None
        resume_learning_rate = 0.0
        resume_amp_scale = 0.0
        resume_optimizer_sha256 = ""
        resume_model_sha256 = ""
        pending_resume: tuple[Path, tuple[int, ...], float, float, str, str] | None
        pending_resume = None
        resume_next_batch_indices: tuple[int, ...] | None = None
        if resume_payload is not None:
            if resume_path is None:
                raise RuntimeError("resume payload is missing its source path")
            payload = _load_checkpoint(
                resume_path,
                route=args.route,
                model=model,
                engine=engine,
                scheduler=scheduler,
                scaler=scaler,
                amp_scale=amp_scale,
                train_loader=train_loader,
                payload=resume_payload,
            )
            resume_next_batch_indices = tuple(payload["next_batch_indices"])
            resume_oracle_path = resume_path.with_suffix(".oracle.json")
            if args.resume_oracle_mode == "require":
                resume_oracle = _load_resume_oracle(resume_oracle_path)
            unwrapped = model.module if hasattr(model, "module") else model
            resume_learning_rate = float(workspace.optimizer.param_groups[0]["lr"])
            resume_amp_scale = amp_scale.value
            resume_optimizer_sha256 = _state_sha256(engine._audit_state())
            resume_model_sha256 = _parameter_sha256(unwrapped)
        effective_start_scale = amp_scale.value
        amp_configuration = _amp_configuration_dict(
            initial_scale=args.amp_initial_scale,
            effective_start_scale=effective_start_scale,
            scaler_state=scaler.state_dict(),
        )
        parity = PairedRouteFacts(
            initial_parameter_sha256=initial_sha256,
            sampler_indices=sampler_indices,
            augmentation_rng_sha256=augmentation_rng_sha256,
            lr_schedule=lr_schedule,
            amp_configuration=_amp_configuration_tuple(amp_configuration),
            batch_size=args.batch_size,
            model_parameter_count=model_parameter_count,
        )
        task_id = f"{args.seed}-{args.route}"
        records: list[dict[str, object]] = []
        epoch_times: list[float] = []
        engine_peak_memory_mib: list[float] = []
        failure_facts: list[dict[str, object]] = []
        validation_total_s = 0.0
        checkpoint_total_s = 0.0
        quality_audit_total_s = 0.0
        data_timer = BatchWaitTimer()
        startup_s = time.perf_counter() - process_started
        planned_steps = total_steps
        if args.max_steps > 0:
            planned_steps = min(planned_steps, args.max_steps)
        production_audit_steps = frozenset(quality_audit_steps(planned_steps))
        validation_loss = 0.0
        raw_path = Path(args.raw_jsonl)
        window_observer = (TrainingLoopWindowObserver(
            lambda: torch.cuda, device, args.warmup_steps
        ) if args.timing_mode == "window" else None)
        if rank == 0:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(epoch_start, args.epochs):
            if (
                epoch == epoch_start
                and replay_batch is not None
                and replay_iterator is not None
                and replay_epoch_indices is not None
            ):
                epoch_indices = replay_epoch_indices
                epoch_batches = chain((replay_batch,), replay_iterator)
                batch_start = 0
            elif epoch == epoch_start and resume_step_in_epoch > 0:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                epoch_indices = _materialize_sampler_indices(
                    train_sampler,
                    len(train_loader),
                )
                resume_offset = resume_step_in_epoch * args.batch_size
                epoch_batches = _iterator_preserving_rng(
                    _resume_loader(train_loader, epoch_indices, epoch=epoch,
                                   start_position=resume_offset)
                )
                batch_start = resume_step_in_epoch
            else:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                epoch_indices = _materialize_sampler_indices(
                    train_sampler,
                    len(train_loader),
                )
                epoch_batches = iter(_resume_loader(train_loader, epoch_indices, epoch=epoch))
                batch_start = 0
            epoch_train_s = 0.0
            stopped_mid_epoch = False
            if window_observer is not None:
                window_observer.begin(epoch=epoch, local_record_start=len(records),
                                      global_step_start=global_step)
            for batch_index, batch in enumerate(
                data_timer.iterate(epoch_batches), start=batch_start
            ):
                start = batch_index * args.batch_size
                batch_indices = tuple(
                    islice(
                        epoch_indices,
                        start,
                        start + args.batch_size,
                    )
                )
                if resume_next_batch_indices:
                    if batch_indices != resume_next_batch_indices:
                        raise ValueError("resume checkpoint next batch drifted")
                    resume_next_batch_indices = None
                torch.cuda.reset_peak_memory_stats(device)
                augmentation_rng = _capture_rng_state()

                def forward() -> object:
                    device_batch = _to_device(batch, device)
                    model_batch = workspace._apply_train_augmentation(device_batch)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return model(model_batch, training=True)

                def backward(loss: object) -> None:
                    scaled_loss = scaler.scale(loss)
                    if args.inject_overflow_step == global_step + 1:
                        scaled_loss = scaled_loss * float("inf")
                    scaled_loss.backward()

                loss, update, forward_s, backward_total_s, engine_total_s = (
                    _PHASE_TIMER.training_step(
                        forward, backward, lambda: engine.step(scaler)
                    )
                )
                epoch_train_s += (
                    float(forward_s) + float(backward_total_s) + float(engine_total_s)
                )
                if not bool(update["skipped"]):
                    scheduler.step()
                global_step += 1
                next_start = (batch_index + 1) * args.batch_size
                next_indices = tuple(
                    islice(
                        epoch_indices,
                        next_start,
                        next_start + args.batch_size,
                    )
                )
                midpoint_checkpoint: Path | None = None
                if global_step == args.smoke_midpoint:
                    midpoint_checkpoint = Path(args.checkpoint_dir)
                    midpoint_checkpoint /= f"midpoint-rank{rank}.pt"
                    checkpoint_started = time.perf_counter()
                    _save_checkpoint(
                        midpoint_checkpoint,
                        route=args.route,
                        epoch=epoch,
                        step=global_step,
                        step_in_epoch=batch_index + 1,
                        next_batch_indices=next_indices,
                        model=model,
                        engine=engine,
                        scheduler=scheduler,
                        scaler=scaler,
                        amp_initial_scale=args.amp_initial_scale,
                        amp_scale=amp_scale,
                        warmup_batch=batch,
                        train_loader=train_loader,
                    )
                    checkpoint_total_s += time.perf_counter() - checkpoint_started
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stopped_mid_epoch = batch_index + 1 < len(train_loader)
                    if stopped_mid_epoch:
                        step_checkpoint = Path(args.checkpoint_dir)
                        step_checkpoint /= f"step-{global_step}-rank{rank}.pt"
                        checkpoint_started = time.perf_counter()
                        _save_checkpoint(
                            step_checkpoint,
                            route=args.route,
                            epoch=epoch,
                            step=global_step,
                            step_in_epoch=batch_index + 1,
                            next_batch_indices=next_indices,
                            model=model,
                            engine=engine,
                            scheduler=scheduler,
                            scaler=scaler,
                            amp_initial_scale=args.amp_initial_scale,
                            amp_scale=amp_scale,
                            warmup_batch=batch,
                            train_loader=train_loader,
                        )
                        checkpoint_total_s += time.perf_counter() - checkpoint_started
                if args.timing_mode != "window":
                    torch.cuda.synchronize(device)
                step_peak_memory_mib = float(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                )
                engine_peak_memory_mib.append(step_peak_memory_mib)
                communication_s = float(update["communication_s"])
                update_communication_s = float(update["update_communication_s"])
                backward_communication_s = max(
                    0.0,
                    communication_s - update_communication_s,
                )
                backward_s = max(0.0, backward_total_s - backward_communication_s)
                update_s = max(0.0, engine_total_s - update_communication_s)
                unwrapped = model.module if hasattr(model, "module") else model
                audit_performed = window_audit_required(
                    quality_audit_mode=args.quality_audit_mode,
                    global_step=global_step, audit_steps=production_audit_steps,
                    final_batch=batch_index + 1 == len(train_loader),
                    midpoint=midpoint_checkpoint is not None,
                    pending_resume=pending_resume is not None,
                    resume_oracle=resume_oracle is not None,
                )
                def observe_quality() -> dict[str, object]:
                    nonlocal quality_audit_total_s
                    quality_start = time.perf_counter()
                    rank_gap: float | None = None
                    model_sha256: str | None = None
                    optimizer_sha256: str | None = None
                    batch_sha256: str | None = None
                    augmentation_sha256: str | None = None
                    if audit_performed:
                        rank_gap = _rank_gap(unwrapped, process_group)
                        model_sha256 = _parameter_sha256(unwrapped)
                        optimizer_sha256 = _state_sha256(engine._audit_state())
                        batch_sha256 = _state_sha256(batch)
                        current_rng = _capture_rng_state()
                        augmentation_sha256 = _reporting_augmentation_sha256(
                            workspace=workspace,
                            batch=batch,
                            device=device,
                            augmentation_rng=augmentation_rng,
                            current_rng=current_rng,
                        )
                    quality_s = time.perf_counter() - quality_start
                    quality_audit_total_s += quality_s
                    return {
                        "rank_gap": rank_gap,
                        "model_sha256": model_sha256,
                        "optimizer_sha256": optimizer_sha256,
                        "batch_sha256": batch_sha256,
                        "augmentation_sha256": augmentation_sha256,
                        "quality_s": quality_s,
                    }
                resumed_facts: ResumeFacts | None = None
                def observation_bookkeeping(
                    loss_value: float | None, observed: dict[str, object]
                ) -> None:
                    nonlocal resume_oracle, pending_resume, resumed_facts
                    batch_sha256 = observed["batch_sha256"]
                    augmentation_sha256 = observed["augmentation_sha256"]
                    optimizer_sha256 = observed["optimizer_sha256"]
                    model_sha256 = observed["model_sha256"]
                    if resume_oracle is not None:
                        resumed_facts = ResumeFacts(
                        next_batch_indices=batch_indices,
                        next_batch_sha256=batch_sha256,
                        next_augmentation_sha256=augmentation_sha256,
                        learning_rate=resume_learning_rate,
                        amp_scale=resume_amp_scale,
                        optimizer_state_sha256=resume_optimizer_sha256,
                        model_sha256=resume_model_sha256,
                        next_loss=loss_value,
                        post_learning_rate=float(
                            workspace.optimizer.param_groups[0]["lr"]
                        ),
                        post_amp_scale=amp_scale.value,
                        post_optimizer_state_sha256=optimizer_sha256,
                        post_model_sha256=model_sha256,
                    )
                        assert_resume_matches(resume_oracle, resumed_facts)
                        resume_oracle = None
                    if pending_resume is not None:
                        (
                        oracle_path,
                        expected_indices,
                        expected_lr,
                        expected_scale,
                        expected_optimizer_sha256,
                        expected_model_sha256,
                        ) = pending_resume
                        if batch_indices != expected_indices:
                            raise ValueError("uninterrupted oracle next batch drifted")
                        _write_resume_oracle(
                        oracle_path,
                        ResumeFacts(
                            next_batch_indices=batch_indices,
                            next_batch_sha256=batch_sha256,
                            next_augmentation_sha256=augmentation_sha256,
                            learning_rate=expected_lr,
                            amp_scale=expected_scale,
                            optimizer_state_sha256=expected_optimizer_sha256,
                            model_sha256=expected_model_sha256,
                            next_loss=loss_value,
                            post_learning_rate=float(
                                workspace.optimizer.param_groups[0]["lr"]
                            ),
                            post_amp_scale=amp_scale.value,
                            post_optimizer_state_sha256=optimizer_sha256,
                            post_model_sha256=model_sha256,
                        ),
                        )
                        pending_resume = None
                def prepare_midpoint_observation(
                    record: dict[str, object], loss_value: float | None,
                    observed: dict[str, object],
                ) -> None:
                    nonlocal pending_resume
                    if midpoint_checkpoint is not None and args.resume_oracle_mode == "write":
                        pending_resume = _prepare_resume_oracle_observation(
                            path=midpoint_checkpoint.with_suffix(".oracle.json"),
                            next_batch_indices=next_indices,
                            learning_rate=float(workspace.optimizer.param_groups[0]["lr"]),
                            amp_scale=amp_scale.value,
                            optimizer_state_sha256=observed["optimizer_sha256"],
                            model_sha256=observed["model_sha256"],
                        )
                def build_observed_record(
                    loss_value: float | None, observed: dict[str, object]
                ) -> dict[str, object]:
                    if bool(update["skipped"]):
                        failure_facts.append(
                            {
                                "phase": "backward_update",
                                "category": "amp_overflow",
                                "message": str(update["decision"]),
                                "rank": rank,
                                "step": global_step,
                                "recoverable": True,
                            }
                        )
                    timing = StepTiming(
                        forward_s=float(forward_s),
                        backward_s=float(backward_s),
                        update_s=float(update_s),
                        communication_s=communication_s,
                        validation_s=0.0,
                        report_serialization_s=float(observed["quality_s"]),
                    )
                    return build_step_record(
                    task_id=task_id,
                    attempt_id=args.attempt_id,
                    route=args.route,
                    seed=args.seed,
                    epoch=epoch,
                    step=global_step,
                    batch_indices=batch_indices,
                    timing=timing,
                    gradient_route=engine.gradient_route,
                    parameter_route=engine.parameter_route,
                    communication_bytes=int(update["communication_bytes"]),
                    qwd_s=float(update["qwd_s"]),
                    refresh_s=float(update["refresh_s"]),
                    decision=str(update["decision"]),
                    loss=loss_value,
                    amp_scale=amp_scale.value,
                    learning_rate=float(workspace.optimizer.param_groups[0]["lr"]),
                    model_sha256=observed["model_sha256"],
                    rank_parameter_gap=observed["rank_gap"],
                    optimizer_step=engine.step_count,
                    finite=not bool(update["skipped"]),
                    audit_performed=audit_performed,
                    defer_loss_validation=args.timing_mode == "window",
                    window_timing=args.timing_mode == "window",
                    )
                record, last_observed = commit_observed_step(
                    timing_mode=args.timing_mode, observer=window_observer,
                    records=records, loss=loss, audit_performed=audit_performed,
                    quality_audit=observe_quality,
                    bookkeeping=observation_bookkeeping,
                    build_record=build_observed_record,
                    post_append=prepare_midpoint_observation, epoch=epoch,
                    global_step=global_step, samples=len(batch_indices),
                    final_batch=batch_index + 1 == len(train_loader),
                    training_stop=args.max_steps > 0 and global_step >= args.max_steps,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            if window_observer is not None:
                window_observer.close("training_stop" if stopped_mid_epoch else "epoch")
            epoch_times.append(epoch_train_s)
            validation_started = time.perf_counter()
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            if args.data_mode == "deterministic":
                val_loader = _resume_loader(val_loader, tuple(val_sampler), epoch=epoch)
            validation_loss = _validate_epoch(model, val_loader, device, args.seed)
            validation_s = time.perf_counter() - validation_started
            validation_total_s += validation_s
            if records and args.timing_mode != "window":
                records[-1]["timing"]["validation_s"] += validation_s
            if not stopped_mid_epoch:
                checkpoint = Path(args.checkpoint_dir)
                checkpoint /= f"epoch-{epoch + 1}-rank{rank}.pt"
                checkpoint_epoch = epoch + 1
                checkpoint_step_in_epoch = 0
                if checkpoint_epoch < args.epochs:
                    if train_sampler is not None:
                        train_sampler.set_epoch(checkpoint_epoch)
                    checkpoint_epoch_indices = _materialize_sampler_indices(
                        train_sampler,
                        len(train_loader),
                    )
                    checkpoint_next_batch_indices = checkpoint_epoch_indices[
                        : args.batch_size
                    ]
                else:
                    checkpoint_next_batch_indices = ()
                checkpoint_started = time.perf_counter()
                _save_checkpoint(
                    checkpoint,
                    route=args.route,
                    epoch=checkpoint_epoch,
                    step=global_step,
                    step_in_epoch=checkpoint_step_in_epoch,
                    next_batch_indices=checkpoint_next_batch_indices,
                    model=model,
                    engine=engine,
                    scheduler=scheduler,
                    scaler=scaler,
                    amp_initial_scale=args.amp_initial_scale,
                    amp_scale=amp_scale,
                    warmup_batch=batch,
                    train_loader=train_loader,
                )
                checkpoint_total_s += time.perf_counter() - checkpoint_started
                if args.resume_oracle_mode == "write" and checkpoint_next_batch_indices:
                    pending_resume = prepare_epoch_resume_oracle_observation(
                        path=checkpoint.with_suffix(".oracle.json"),
                        next_batch_indices=checkpoint_next_batch_indices,
                        learning_rate=float(workspace.optimizer.param_groups[0]["lr"]),
                        amp_scale=amp_scale.value,
                        observed=last_observed,
                    )
            resume_step_in_epoch = 0
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        local_telemetry = gpu_sampler.stop()
        gpu_sampler = None
        gathered_telemetry: list[object] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gathered_telemetry,
            local_telemetry,
            group=process_group,
        )
        raw_started = time.perf_counter()
        rank_raw_path = (
            raw_path
            if rank == 0
            else raw_path.with_name(f"{raw_path.stem}.rank{rank}{raw_path.suffix}")
        )
        _write_raw_records(rank_raw_path, records)
        raw_report_s = time.perf_counter() - raw_started
        local_execution = {
            "rank": rank,
            **execution_counters(records),
            "initial_sampler_indices_sha256": canonical_sha256(sampler_indices),
            "initial_sampler_numel": len(sampler_indices),
            "data_wait_s": data_timer.wait_s,
            "raw_file": str(rank_raw_path),
            "raw_sha256": sha256(rank_raw_path.read_bytes()).hexdigest(),
        }
        local_execution.update(window_sidecar_fields(window_observer))
        gathered_execution: list[object] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gathered_execution, local_execution, group=process_group
        )
        if rank == 0:
            if not records:
                raise RuntimeError("training produced no step records")
            report_started = time.perf_counter()
            summary = (summarize_window_records(tuple(records)) if args.timing_mode == "window" else summarize_step_records(
                tuple(records),
                warmup_steps=args.warmup_steps,
                batch_size_per_rank=args.batch_size,
                world_size=world_size,
            ))
            if args.timing_mode == "window":
                validate_window_rank_coverage(gathered_execution)
            manifest = source_tree_manifest(args.psi_source)
            physical = tuple(int(item["physical_gpu_index"]) for item in rank_devices)
            report_s = raw_report_s + time.perf_counter() - report_started
            observed_process_s = time.perf_counter() - process_started
            core_train_s = float(sum(epoch_times))
            known_process_s = (
                startup_s
                + data_timer.wait_s
                + core_train_s
                + validation_total_s
                + checkpoint_total_s
                + quality_audit_total_s
                + report_s
            )
            other_s = max(0.0, observed_process_s - known_process_s)
            process_wall_s = observed_process_wall(args.timing_mode, observed_process_s,
                                                   known_process_s + other_s)
            result = build_task_result(
                task_id=task_id,
                attempt_id=args.attempt_id,
                route=args.route,
                seed=args.seed,
                world_size=world_size,
                physical_gpu_ids=physical,
                rank_devices=rank_devices,
                source_manifest_sha256=str(manifest["manifest_sha256"]),
                data_sha256=args.data_sha256,
                parity=parity,
                epochs=len(epoch_times),
                steps=len(records),
                warmup_steps=args.warmup_steps,
                steady_samples_per_second=float(summary.get("steady_samples_per_second", 0.0)),
                step_latency_p50_ms=float(summary.get("step_latency_p50_ms", 0.0)),
                step_latency_p95_ms=float(summary.get("step_latency_p95_ms", 0.0)),
                epoch_core_time_s=tuple(epoch_times),
                timing_breakdown={
                    "startup_s": float(startup_s),
                    "data_s": float(data_timer.wait_s),
                    "core_train_s": core_train_s,
                    "validation_s": float(validation_total_s),
                    "checkpoint_s": float(checkpoint_total_s),
                    "quality_audit_s": float(quality_audit_total_s),
                    "report_s": float(report_s),
                    "other_s": float(other_s),
                    "process_wall_s": float(process_wall_s),
                },
                communication_time_s=float(summary["communication_time_s"]),
                qwd_time_s=float(summary["qwd_time_s"]),
                refresh_time_s=float(summary["refresh_time_s"]),
                communication_bytes=int(summary["communication_bytes"]),
                peak_memory_mib=max(engine_peak_memory_mib),
                gpu_telemetry=tuple(dict(item) for item in gathered_telemetry),
                loss_trajectory=tuple(summary["loss_trajectory"]),
                validation_loss=float(validation_loss),
                rank_gaps=tuple(summary["rank_gaps"]),
                decision_counts=summary["decision_counts"],
                failure_facts=tuple(failure_facts),
                execution_protocol=engine.training_protocol,
                training_loop_windows=(tuple(window_observer.windows) if window_observer is not None else None),
                local_record_samples=(tuple(len(record["batch_indices"]) for record in records)
                                      if window_observer is not None else None),
            )
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(result, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # This direct/private-plan diagnostic must never impersonate an
            # adapter qualification or an independently validated data split.
            evidence = {
                "schema_version": 1,
                "training_protocol": engine.training_protocol,
                "qualification_eligible": False,
                "validation_split_independence": "not_verified",
                **measurement_observability(
                    args.timing_mode, args.route, args.native_ddp_mode, args.model_precision
                ),
                "timing_rank": 0,
                "communication_bytes_semantics": "route_specific_estimate_not_measured_wire_bytes",
                "rank_execution": gathered_execution,
                "torch_version": str(torch.__version__),
                "worker_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_sha256": sha256(
                    Path(__file__).with_name("psi_training_runtime.py").read_bytes()
                ).hexdigest(),
                "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
            }
            result_path.with_suffix(".protocol.json").write_text(
                json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if gpu_sampler is not None:
            gpu_sampler.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    """Run one exact CLI-selected distributed PSI route."""
    args = parse_args()
    placement = parse_cuda_process_placement(
        args.cpu_affinity_map,
        args.nccl_channels,
    )
    apply_cuda_process_placement(
        placement,
        local_rank=int(os.environ["LOCAL_RANK"]),
        local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
    )
    _run(args)


if __name__ == "__main__":
    main()
