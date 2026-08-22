"""One distributed worker for Native, CAG, and RSAG/qWD PSI training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from types import ModuleType
from tests.benchmarks.psi_v040_state import QWDSchedule, ShardLayout, ShardedAdamW
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
    validate_step_record,
)


_CACHE_PARTS = frozenset({"__pycache__", ".pytest_cache", "__MACOSX"})
_DDP_BUCKET_WARMUP_BACKWARDS = 2
_LOCKED_OVERRIDE_KEYS = frozenset(
    {
        "communication.enabled",
        "logging.mode",
        "notifications.feishu.enabled",
        "train_dataloader.batch_size",
        "training.gradient_accumulate_every",
        "training.num_epochs",
        "training.seed",
        "training.use_ema",
    }
)


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
        "rank_gaps": tuple(float(value["rank_parameter_gap"]) for value in qualities),
        "decision_counts": dict(counts),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _cuda_timed(action: Callable[[], object]) -> tuple[object, float]:
    torch = _torch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = action()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end)) / 1000.0


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

    def __init__(
        self,
        *,
        model: object,
        optimizer: object,
        grad_clip: float,
        telemetry: _HookTelemetry,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.telemetry = telemetry
        self.step_count = 0

    def step(self, scaler: object) -> dict[str, object]:
        torch = _torch()
        overflow, _, overflow_control_s, overflow_control_bytes = (
            _unscale_and_detect_overflow(
            tuple(self.model.parameters()),
            scaler,
            torch.distributed.group.WORLD,
        )
        )
        def update_optimizer() -> None:
            if not overflow:
                torch.nn.utils.clip_grad_norm_(
                    tuple(self.model.parameters()),
                    self.grad_clip,
                )
                self.optimizer.step()

        _, update_s = _cuda_timed(update_optimizer)
        _advance_amp_scaler(scaler, overflow=overflow)
        skipped = overflow
        self.optimizer.zero_grad(set_to_none=True)
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
            "decision": (
                "overflow_after_communication" if skipped else self.route
            ),
            "skipped": skipped,
            "update_communication_s": overflow_control_s,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        _require_fields(state, {"optimizer", "step_count"}, "native state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state["step_count"])


class CAGUpdateEngine(NativeUpdateEngine):
    """Full AdamW after v0.4 FullTensor transactional-EF CAG."""

    route = "cag"
    gradient_route = "fulltensor_int8_group64_ef"

    def step(self, scaler: object) -> dict[str, object]:
        result = super().step(scaler)
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "gradient_feedback": self.telemetry.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fields = {"optimizer", "step_count", "gradient_feedback"}
        _require_fields(state, fields, "CAG state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state["step_count"])
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
    ) -> None:
        torch = _torch()
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not self.parameters:
            raise RuntimeError("RSAG/qWD requires trainable parameters")
        self.global_numel = sum(parameter.numel() for parameter in self.parameters)
        self.layout = ShardLayout.build(self.global_numel, world_size, rank)
        self.padded_model_numel = self.layout.padded_numel * world_size
        flat = self._flat_model()
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
        self.schedule = QWDSchedule(refresh_interval=100)
        self.gradient_plan = _build_cuda_plan(
            output="reduced_shard",
            numel=self.global_numel,
            rank=rank,
            world_size=world_size,
            process_group=process_group,
        )
        extension = import_module("lowbit_comm._C")
        self.qwd_config = _qwd_config(self.global_numel, rank, world_size)
        self.qwd_plan = extension._create_qwd_plan(
            self.qwd_config,
            process_group,
        )

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
            scaler,
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
            flat_gradient = (
                torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
                .to(dtype=torch.float16, copy=False)
                .contiguous()
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
                clip = min(1.0, self.grad_clip / (float(norm_sq.sqrt()) + 1.0e-6))
                reduced_shard.mul_(clip)
                return reduced_shard

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

            try:
                _, parameter_s = _cuda_timed(commit_parameter_update)
            except Exception:
                self.gradient_plan._restore_committed_residual(feedback_before)
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
        _advance_amp_scaler(scaler, overflow=overflow)
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
        torch = _torch()
        candidate = self.candidate_optimizer
        candidate.master.copy_(self.master)
        candidate.exp_avg.copy_(self.exp_avg)
        candidate.exp_avg_sq.copy_(self.exp_avg_sq)
        candidate.step_count = self.step_count
        valid = self.layout.valid_numel
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        with torch.no_grad():
            if valid:
                candidate.master[:valid].mul_(
                    1.0 - learning_rate * self.weight_decay[:valid]
                )
        candidate.learning_rate = learning_rate
        candidate.step_prevalidated(reduced_shard)
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
        gradient = int(self.gradient_plan.layout.send_payload_bytes)
        gradient += int(self.gradient_plan.layout.receive_payload_bytes)
        if mode == "qwd":
            parameter = int(self.qwd_config["qwd_gathered_payload_bytes"])
        else:
            parameter = int(self.qwd_config["fp32_gathered_bytes"])
        return gradient + parameter

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
                "key": ("rsag", self.rank, self.world_size, self.global_numel),
                "layout": _plan_layout_facts(self.gradient_plan),
                "residual": _clone_plan_feedback(self.gradient_plan),
            },
            "learning_rates": tuple(
                float(group["lr"]) for group in self.optimizer.param_groups
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fields = {
            "layout",
            "optimizer",
            "gradient_feedback",
            "learning_rates",
        }
        _require_fields(state, fields, "RSAG/qWD state")
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
        expected_key = ("rsag", self.rank, self.world_size, self.global_numel)
        _restore_plan_feedback(
            self.gradient_plan,
            state["gradient_feedback"],
            expected_key=expected_key,
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
        self.force_refresh = True


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


def _qwd_config(
    global_numel: int,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    shard_numel = (global_numel + world_size - 1) // world_size
    start = min(rank * shard_numel, global_numel)
    valid_numel = min(shard_numel, global_numel - start)
    groups = (shard_numel + 63) // 64
    payload = groups * 68
    gathered_payload = payload * world_size
    fp32_gathered = shard_numel * world_size * 4
    return {
        "accumulation_dtype": "fp32",
        "collective": "all_gather",
        "compression": "int8",
        "dtype": "fp16",
        "fp32_gathered_bytes": fp32_gathered,
        "global_numel": global_numel,
        "group_size": 64,
        "groups_per_shard": groups,
        "output_bytes": shard_numel * world_size * 2,
        "payload_bytes_per_rank": payload,
        "qwd_gathered_payload_bytes": gathered_payload,
        "rank": rank,
        "shard_numel": shard_numel,
        "start": start,
        "valid_numel": valid_numel,
        "workspace_bytes": max(payload + gathered_payload, fp32_gathered),
        "world_size": world_size,
    }


def _plan_layout_facts(plan: object) -> tuple[tuple[str, object], ...]:
    layout = plan.layout
    return tuple((item.name, getattr(layout, item.name)) for item in fields(layout))


def _clone_plan_feedback(plan: object) -> object | None:
    residual = plan._committed_residual
    return None if residual is None else residual.detach().clone()


def _restore_plan_feedback(
    plan: object,
    state: object,
    *,
    expected_key: tuple[object, ...],
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
            or residual.device.type != "cuda"
            or residual.dtype is not expected_dtype
            or residual.ndim != 1
            or residual.numel() != plan.intent.tensor.numel
            or not residual.is_contiguous()
        ):
            raise ValueError("gradient feedback residual is inconsistent")
    plan._restore_committed_residual(residual)


def _register_ddp_hook(model: object, route: str) -> _HookTelemetry:
    torch = _torch()
    telemetry = _HookTelemetry()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def hook(_: object, bucket: object) -> object:
        buffer = bucket.buffer()
        if route == "native":
            def communicate() -> None:
                torch.distributed.all_reduce(buffer)
                buffer.div_(world_size)

            _, elapsed_s = _cuda_timed(communicate)
            byte_count = buffer.numel() * buffer.element_size() * 2
            byte_count *= world_size - 1
        else:
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
            bucket_key = (
                int(bucket.index()),
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
            telemetry.snapshot_feedback(bucket_key)
            def communicate() -> None:
                buffer.copy_(plan.execute(compressed).wait())

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
    scaler: object,
    process_group: object,
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
        1.0 / float(scaler.get_scale()),
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
    return bool(found_inf.item()), gradients, communication_s, communication_bytes


def _advance_amp_scaler(scaler: object, *, overflow: bool) -> None:
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
    sys.dont_write_bytecode = True
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import_module("psi_policy.train")
    _install_psi_update_seams()
    return import_module("psi_policy.workspace.train_workspace")


def _install_psi_update_seams() -> None:
    """Block legacy PSI engines while retaining the immutable workspace."""
    parent = import_module("psi_policy.communication")

    def worker_owned(*_: object, **__: object) -> object:
        raise RuntimeError("Task 5 worker owns the distributed update engine")

    modules = {
        "ccdl_ddp": ("register_ccdl_ddp_hook",),
        "ccdl_sharded_adamw": ("prepare_psi_sharded_adamw",),
    }
    for leaf, functions in modules.items():
        name = f"psi_policy.communication.{leaf}"
        module = ModuleType(name)
        for function in functions:
            setattr(module, function, worker_owned)
        sys.modules[name] = module
        setattr(parent, leaf, module)


def _build_workspace(args: object, rank: int, world_size: int) -> tuple[object, ...]:
    source = Path(args.psi_source).resolve()
    workspace_module = _import_psi_source(source)
    hydra = import_module("hydra")
    _reject_locked_psi_overrides(args.psi_override)
    overrides = [
        f"training.seed={args.seed}",
        f"training.num_epochs={args.epochs}",
        f"train_dataloader.batch_size={args.batch_size}",
        "training.gradient_accumulate_every=1",
        "training.use_ema=false",
        "communication.enabled=false",
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
    return workspace, train_loader, train_sampler, val_loader, val_sampler


def _reject_locked_psi_overrides(overrides: object) -> None:
    if type(overrides) is not list or not all(type(value) is str for value in overrides):
        raise ValueError("PSI overrides must be an exact string list")
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in _LOCKED_OVERRIDE_KEYS:
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
    torch = _torch()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=torch.float16)


def _object_sha256(value: object) -> str:
    torch = _torch()
    buffer = BytesIO()
    torch.save(value, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def _state_sha256(value: object) -> str:
    """Hash nested optimizer state independently of pickle storage identities."""
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
        if type(item) is dict:
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


def _materialize_sampler_indices(
    sampler: object, loader_length: int
) -> tuple[int, ...]:
    if sampler is None:
        return tuple(range(loader_length))
    return tuple(int(value) for value in sampler)


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
    facts = []
    for position, line in enumerate(output.splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or position >= len(physical_gpu_ids):
            continue
        facts.append(
            {
                "gpu": physical_gpu_ids[position],
                "utilization": float(fields[1]),
                "memory_used_mib": float(fields[2]),
                "temperature_c": float(fields[3]),
                "sm_clock_mhz": float(fields[4]),
            }
        )
    return tuple(facts)


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
) -> None:
    torch = _torch()
    payload = {
        "route": route,
        "epoch": epoch,
        "step": step,
        "step_in_epoch": step_in_epoch,
        "next_batch_indices": next_batch_indices,
        "model": model.state_dict(),
        "engine": engine.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
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
) -> dict[str, object]:
    torch = _torch()
    payload = torch.load(path, map_location="cuda", weights_only=False)
    fields = {
        "route",
        "epoch",
        "step",
        "step_in_epoch",
        "next_batch_indices",
        "model",
        "engine",
        "scheduler",
        "scaler",
        "rng",
    }
    _require_fields(payload, fields, "checkpoint")
    if payload["route"] != route:
        raise ValueError("checkpoint route is inconsistent")
    model.load_state_dict(payload["model"])
    engine.load_state_dict(payload["engine"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random = import_module("random")
    numpy = import_module("numpy")
    random.setstate(payload["rng"]["python"])
    numpy.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng"]["cuda"]])
    return payload


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


def _load_resume_oracle(path: Path) -> ResumeFacts:
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "next_batch_indices",
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
    """Validate and account canonical preparation before publishing raw rows."""
    lines = []
    for record in records:
        started = time.perf_counter()
        validate_step_record(record)
        json.dumps(record, sort_keys=True)
        elapsed_s = time.perf_counter() - started
        record["timing"]["report_serialization_s"] += elapsed_s
        validate_step_record(record)
        lines.append(json.dumps(record, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_epoch(model: object, loader: object, device: object) -> float:
    torch = _torch()
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
    model.train()
    if sample_count.item() == 0.0:
        return 0.0
    return float((loss_sum / sample_count).item())


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
    """Keep DDP's initial and rebuilt layouts to one trainable-parameter bucket."""
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
    train_loader: object,
    device: object,
    telemetry: _HookTelemetry,
) -> None:
    """Learn DDP's final bucket order without changing training state."""
    torch = _torch()
    random = import_module("random")
    numpy = import_module("numpy")
    rng = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    buffers = tuple(
        (buffer, buffer.detach().clone()) for buffer in model.buffers()
    )
    training = bool(model.training)
    try:
        batch = next(iter(train_loader))
        for _ in range(_DDP_BUCKET_WARMUP_BACKWARDS):
            device_batch = _to_device(batch, device)
            model_batch = workspace._apply_train_augmentation(device_batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(model_batch, training=True)
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        telemetry.reset_after_warmup()
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        model.train(training)
        random.setstate(rng["python"])
        numpy.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])


def _run(args: object) -> None:
    torch = _torch()
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    process_group = torch.distributed.group.WORLD
    try:
        workspace, train_loader, train_sampler, val_loader, val_sampler = (
            _build_workspace(args, rank, world_size)
        )
        del val_sampler
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
        _convert_model_to_common_fp16(model)
        ddp_bucket_cap_mb = _stable_ddp_bucket_cap_mb(model)
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
        )
        if args.route in {"native", "cag"}:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                bucket_cap_mb=ddp_bucket_cap_mb,
            )
            telemetry = _register_ddp_hook(model, args.route)
            _stabilize_ddp_bucket_layout(
                model,
                workspace,
                train_loader,
                device,
                telemetry,
            )
        else:
            telemetry = _HookTelemetry()
        engine = build_engine(
            args.route,
            native_factory=lambda: NativeUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                telemetry=telemetry,
            ),
            cag_factory=lambda: CAGUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                telemetry=telemetry,
            ),
            rsag_qwd_factory=lambda: RSAGQWDUpdateEngine(
                model=model,
                optimizer=workspace.optimizer,
                grad_clip=float(workspace.cfg.training.grad_norm_clip),
                rank=rank,
                world_size=world_size,
                process_group=process_group,
            ),
        )
        unwrapped = model.module if hasattr(model, "module") else model
        initial_sha256 = _parameter_sha256(unwrapped)
        parity = PairedRouteFacts(
            initial_parameter_sha256=initial_sha256,
            sampler_indices=sampler_indices,
            augmentation_rng_sha256=augmentation_rng_sha256,
            lr_schedule=lr_schedule,
            amp_configuration=("fp16", True, float(scaler.get_scale())),
            batch_size=args.batch_size,
            model_parameter_count=model_parameter_count,
        )
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
        epoch_start = 0
        global_step = 0
        resume_step_in_epoch = 0
        resume_oracle: ResumeFacts | None = None
        resume_oracle_path: Path | None = None
        resume_learning_rate = 0.0
        resume_amp_scale = 0.0
        resume_optimizer_sha256 = ""
        resume_model_sha256 = ""
        pending_resume: tuple[Path, tuple[int, ...], float, float, str, str] | None
        pending_resume = None
        if args.inject_overflow_step < 0:
            raise ValueError("inject_overflow_step must be non-negative")
        if args.resume_oracle_mode == "require" and args.resume is None:
            raise ValueError("resume oracle require mode needs --resume")
        if args.resume_oracle_mode == "write" and args.resume is not None:
            raise ValueError("resume oracle write mode requires an uninterrupted run")
        if args.resume is not None:
            resume_path = _resolve_resume_path(args.resume, rank)
            payload = _load_checkpoint(
                resume_path,
                route=args.route,
                model=model,
                engine=engine,
                scheduler=scheduler,
                scaler=scaler,
            )
            epoch_start = int(payload["epoch"])
            global_step = int(payload["step"])
            resume_step_in_epoch = int(payload["step_in_epoch"])
            resume_oracle_path = resume_path.with_suffix(".oracle.json")
            if args.resume_oracle_mode == "require":
                resume_oracle = _load_resume_oracle(resume_oracle_path)
            unwrapped = model.module if hasattr(model, "module") else model
            resume_learning_rate = float(workspace.optimizer.param_groups[0]["lr"])
            resume_amp_scale = float(scaler.get_scale())
            resume_optimizer_sha256 = _state_sha256(engine.state_dict())
            resume_model_sha256 = _parameter_sha256(unwrapped)
        task_id = f"{args.seed}-{args.route}"
        records: list[dict[str, object]] = []
        epoch_times: list[float] = []
        engine_peak_memory_mib: list[float] = []
        failure_facts: list[dict[str, object]] = []
        validation_loss = 0.0
        raw_path = Path(args.raw_jsonl)
        if rank == 0:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(epoch_start, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_indices = _materialize_sampler_indices(
                train_sampler,
                len(train_loader),
            )
            epoch_started = time.perf_counter()
            for batch_index, batch in enumerate(train_loader):
                if epoch == epoch_start and batch_index < resume_step_in_epoch:
                    continue
                start = batch_index * args.batch_size
                batch_indices = tuple(
                    islice(
                        epoch_indices,
                        start,
                        start + args.batch_size,
                    )
                )
                torch.cuda.reset_peak_memory_stats(device)

                def forward() -> object:
                    device_batch = _to_device(batch, device)
                    model_batch = workspace._apply_train_augmentation(device_batch)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return model(model_batch, training=True)

                loss, forward_s = _cuda_timed(forward)

                def backward() -> None:
                    scaled_loss = scaler.scale(loss)
                    if args.inject_overflow_step == global_step + 1:
                        scaled_loss = scaled_loss * float("inf")
                    scaled_loss.backward()

                _, backward_total_s = _cuda_timed(backward)
                update, engine_total_s = _cuda_timed(lambda: engine.step(scaler))
                if not bool(update["skipped"]):
                    scheduler.step()
                global_step += 1
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
                quality_start = time.perf_counter()
                unwrapped = model.module if hasattr(model, "module") else model
                rank_gap = _rank_gap(unwrapped, process_group)
                model_sha256 = _parameter_sha256(unwrapped)
                optimizer_sha256 = _state_sha256(engine.state_dict())
                loss_value = float(loss.detach())
                quality_s = time.perf_counter() - quality_start
                resumed_facts: ResumeFacts | None = None
                if resume_oracle is not None:
                    resumed_facts = ResumeFacts(
                        next_batch_indices=batch_indices,
                        learning_rate=resume_learning_rate,
                        amp_scale=resume_amp_scale,
                        optimizer_state_sha256=resume_optimizer_sha256,
                        model_sha256=resume_model_sha256,
                        next_loss=loss_value,
                        post_learning_rate=float(
                            workspace.optimizer.param_groups[0]["lr"]
                        ),
                        post_amp_scale=float(scaler.get_scale()),
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
                            learning_rate=expected_lr,
                            amp_scale=expected_scale,
                            optimizer_state_sha256=expected_optimizer_sha256,
                            model_sha256=expected_model_sha256,
                            next_loss=loss_value,
                            post_learning_rate=float(
                                workspace.optimizer.param_groups[0]["lr"]
                            ),
                            post_amp_scale=float(scaler.get_scale()),
                            post_optimizer_state_sha256=optimizer_sha256,
                            post_model_sha256=model_sha256,
                        ),
                    )
                    pending_resume = None
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
                    report_serialization_s=float(quality_s),
                )
                record = build_step_record(
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
                    amp_scale=float(scaler.get_scale()),
                    learning_rate=float(workspace.optimizer.param_groups[0]["lr"]),
                    model_sha256=model_sha256,
                    rank_parameter_gap=rank_gap,
                    optimizer_step=engine.step_count,
                    finite=not bool(update["skipped"]),
                )
                records.append(record)
                if global_step == args.smoke_midpoint:
                    next_start = (batch_index + 1) * args.batch_size
                    next_indices = tuple(
                        islice(
                            epoch_indices,
                            next_start,
                            next_start + args.batch_size,
                        )
                    )
                    checkpoint = Path(args.checkpoint_dir)
                    checkpoint /= f"midpoint-rank{rank}.pt"
                    _save_checkpoint(
                        checkpoint,
                        route=args.route,
                        epoch=epoch,
                        step=global_step,
                        step_in_epoch=batch_index + 1,
                        next_batch_indices=next_indices,
                        model=model,
                        engine=engine,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    if args.resume_oracle_mode == "write":
                        pending_resume = (
                            checkpoint.with_suffix(".oracle.json"),
                            next_indices,
                            float(workspace.optimizer.param_groups[0]["lr"]),
                            float(scaler.get_scale()),
                            _state_sha256(engine.state_dict()),
                            model_sha256,
                        )
                        if isinstance(engine, RSAGQWDUpdateEngine):
                            engine.force_refresh = True
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            epoch_times.append(time.perf_counter() - epoch_started)
            validation_started = time.perf_counter()
            validation_loss = _validate_epoch(model, val_loader, device)
            validation_s = time.perf_counter() - validation_started
            if records:
                records[-1]["timing"]["validation_s"] += validation_s
            checkpoint = Path(args.checkpoint_dir)
            checkpoint /= f"epoch-{epoch + 1}-rank{rank}.pt"
            _save_checkpoint(
                checkpoint,
                route=args.route,
                epoch=epoch + 1,
                step=global_step,
                step_in_epoch=0,
                next_batch_indices=(),
                model=model,
                engine=engine,
                scheduler=scheduler,
                scaler=scaler,
            )
            resume_step_in_epoch = 0
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if rank == 0:
            if not records:
                raise RuntimeError("training produced no step records")
            _write_raw_records(raw_path, records)
            summary = summarize_step_records(
                tuple(records),
                warmup_steps=args.warmup_steps,
                batch_size_per_rank=args.batch_size,
                world_size=world_size,
            )
            manifest = source_tree_manifest(args.psi_source)
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not visible_devices:
                visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
            try:
                physical = tuple(
                    int(value) for value in visible_devices.split(",") if value.strip()
                )
            except ValueError:
                physical = ()
            if len(physical) != world_size:
                physical = tuple(range(world_size))
            result = build_task_result(
                task_id=task_id,
                attempt_id=args.attempt_id,
                route=args.route,
                seed=args.seed,
                world_size=world_size,
                physical_gpu_ids=physical,
                source_manifest_sha256=str(manifest["manifest_sha256"]),
                data_sha256=args.data_sha256,
                parity=parity,
                epochs=len(epoch_times),
                steps=len(records),
                warmup_steps=args.warmup_steps,
                steady_samples_per_second=float(summary["steady_samples_per_second"]),
                step_latency_p50_ms=float(summary["step_latency_p50_ms"]),
                step_latency_p95_ms=float(summary["step_latency_p95_ms"]),
                epoch_time_s=tuple(epoch_times),
                communication_time_s=float(summary["communication_time_s"]),
                qwd_time_s=float(summary["qwd_time_s"]),
                refresh_time_s=float(summary["refresh_time_s"]),
                communication_bytes=int(summary["communication_bytes"]),
                peak_memory_mib=max(engine_peak_memory_mib),
                gpu_telemetry=_gpu_telemetry(physical),
                loss_trajectory=tuple(summary["loss_trajectory"]),
                validation_loss=float(validation_loss),
                rank_gaps=tuple(summary["rank_gaps"]),
                decision_counts=summary["decision_counts"],
                failure_facts=tuple(failure_facts),
            )
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(result, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    """Run one exact CLI-selected distributed PSI route."""
    _run(parse_args())


if __name__ == "__main__":
    main()
