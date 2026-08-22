"""One distributed worker for Native, CAG, and RSAG/qWD PSI training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path
from statistics import median
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


@dataclass(slots=True)
class _HookTelemetry:
    communication_s: float = 0.0
    communication_bytes: int = 0

    def add(self, elapsed_s: float, byte_count: int) -> None:
        self.communication_s += elapsed_s
        self.communication_bytes += byte_count

    def consume(self) -> tuple[float, int]:
        result = self.communication_s, self.communication_bytes
        self.communication_s = 0.0
        self.communication_bytes = 0
        return result


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
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            tuple(self.model.parameters()),
            self.grad_clip,
        )
        start = time.perf_counter()
        scale = float(scaler.get_scale())
        scaler.step(self.optimizer)
        scaler.update()
        skipped = float(scaler.get_scale()) < scale
        update_s = time.perf_counter() - start
        self.optimizer.zero_grad(set_to_none=True)
        if not skipped:
            self.step_count += 1
        communication_s, communication_bytes = self.telemetry.consume()
        return {
            "update_s": update_s,
            "communication_s": communication_s,
            "communication_bytes": communication_bytes,
            "qwd_s": 0.0,
            "refresh_s": 0.0,
            "decision": "native",
            "skipped": skipped,
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
        result["decision"] = "cag"
        return result


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
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.data = parameter.data.to(dtype=torch.float16)
        self.global_numel = sum(parameter.numel() for parameter in self.parameters)
        self.layout = ShardLayout.build(self.global_numel, world_size, rank)
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
        self.weight_decay = self._weight_decay_shard()
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
        update_started = time.perf_counter()
        gradients = tuple(parameter.grad for parameter in self.parameters)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("every RSAG/qWD parameter requires a gradient")
        _unscale_fp16_gradients(gradients, float(scaler.get_scale()))
        found_inf = torch.tensor(
            [
                0.0
                if all(torch.isfinite(gradient).all() for gradient in gradients)
                else 1.0
            ],
            device=self.parameters[0].device,
        )
        torch.distributed.all_reduce(
            found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=self.process_group,
        )
        if found_inf.item() != 0.0:
            scale = float(scaler.get_scale())
            scaler.update(new_scale=max(1.0, scale / 2.0))
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "update_s": time.perf_counter() - update_started,
                "communication_s": 0.0,
                "communication_bytes": 0,
                "qwd_s": 0.0,
                "refresh_s": 0.0,
                "decision": "overflow_skip",
                "skipped": True,
            }
        flat_gradient = (
            torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
            .to(dtype=torch.float16, copy=False)
            .contiguous()
        )
        communication_start = time.perf_counter()
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
        communication_s = time.perf_counter() - communication_start
        update_start = time.perf_counter()
        candidate = self._adamw_candidate(reduced_shard)
        mode = self.schedule.mode(
            self.step_count,
            force_refresh=self.force_refresh,
        )
        flat_model = self._flat_model()
        parameter_start = time.perf_counter()
        work = self.qwd_plan.execute(candidate.master, flat_model, mode)
        transaction = RSAGQWDTransaction(
            publish_optimizer=self._publish_optimizer,
            publish_model=self._publish_model,
        )
        transaction.commit(candidate, work)
        parameter_s = time.perf_counter() - parameter_start
        self.force_refresh = False
        scaler.update(new_scale=float(scaler.get_scale()))
        self.optimizer.zero_grad(set_to_none=True)
        communication_s += parameter_s
        return {
            "update_s": max(0.0, time.perf_counter() - update_start - parameter_s),
            "communication_s": communication_s,
            "communication_bytes": self._communication_bytes(mode),
            "qwd_s": parameter_s if mode == "qwd" else 0.0,
            "refresh_s": parameter_s if mode == "fp_refresh" else 0.0,
            "decision": mode,
            "skipped": False,
        }

    def _flat_model(self) -> object:
        torch = _torch()
        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self.parameters]
        ).contiguous()

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
        candidate = ShardedAdamW(
            self.layout,
            self.master,
            learning_rate=self.base_learning_rate,
            betas=self.sharded_optimizer.betas,
            eps=self.sharded_optimizer.eps,
            weight_decay=0.0,
        )
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
        candidate.step(reduced_shard)
        candidate.learning_rate = max(learning_rate, self.base_learning_rate)
        return candidate

    def _publish_optimizer(self, candidate: object) -> None:
        if type(candidate) is not ShardedAdamW:
            raise ValueError("optimizer candidate must be exact ShardedAdamW")
        self.sharded_optimizer = candidate

    def _publish_model(self, flat: object) -> None:
        torch = _torch()
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
            "learning_rates": tuple(
                float(group["lr"]) for group in self.optimizer.param_groups
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        fields = {
            "layout",
            "optimizer",
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


def _register_ddp_hook(model: object, route: str) -> _HookTelemetry:
    torch = _torch()
    telemetry = _HookTelemetry()
    plans: dict[int, object] = {}
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def hook(_: object, bucket: object) -> object:
        buffer = bucket.buffer()
        start = time.perf_counter()
        if route == "native":
            torch.distributed.all_reduce(buffer)
            buffer.div_(world_size)
            byte_count = buffer.numel() * buffer.element_size() * 2
            byte_count *= world_size - 1
        else:
            compressed = buffer.to(dtype=torch.float16).contiguous()
            plan = plans.get(buffer.numel())
            if plan is None:
                plan = _build_cuda_plan(
                    output="fulltensor",
                    numel=buffer.numel(),
                    rank=rank,
                    world_size=world_size,
                    process_group=torch.distributed.group.WORLD,
                )
                plans[buffer.numel()] = plan
            buffer.copy_(plan.execute(compressed).wait())
            byte_count = int(plan.layout.gathered_payload_bytes)
        telemetry.add(time.perf_counter() - start, byte_count)
        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    hook.__annotations__["bucket"] = torch.distributed.GradBucket
    hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]
    model.register_comm_hook(state=None, hook=hook)
    return telemetry


def _torch() -> object:
    return import_module("torch")


def _unscale_fp16_gradients(
    gradients: tuple[object, ...],
    scale: float,
) -> None:
    if scale <= 0.0:
        raise ValueError("AMP scale must be positive")
    inverse = 1.0 / scale
    for gradient in gradients:
        gradient.mul_(inverse)


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


def _parameter_sha256(model: object) -> str:
    digest = sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


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


def _gpu_telemetry() -> tuple[dict[str, object], ...]:
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
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        facts.append(
            {
                "gpu": int(fields[0]),
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
    }
    _require_fields(value, fields, "resume oracle")
    return ResumeFacts(
        next_batch_indices=tuple(value["next_batch_indices"]),
        learning_rate=value["learning_rate"],
        amp_scale=value["amp_scale"],
        optimizer_state_sha256=value["optimizer_state_sha256"],
        model_sha256=value["model_sha256"],
        next_loss=value["next_loss"],
    )


def _validate_epoch(model: object, loader: object, device: object) -> float:
    torch = _torch()
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            value = _to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(value, training=True)
            losses.append(float(loss))
    model.train()
    return float(median(losses)) if losses else 0.0


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
        initial_sha256 = _parameter_sha256(workspace.model)
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
        model = workspace.model
        if args.route in {"native", "cag"}:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            telemetry = _register_ddp_hook(model, args.route)
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
        resume_learning_rate = 0.0
        resume_amp_scale = 0.0
        resume_optimizer_sha256 = ""
        resume_model_sha256 = ""
        pending_resume: tuple[Path, tuple[int, ...], float, float, str, str] | None
        pending_resume = None
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
            resume_oracle = _load_resume_oracle(resume_path.with_suffix(".oracle.json"))
            unwrapped = model.module if hasattr(model, "module") else model
            resume_learning_rate = float(workspace.optimizer.param_groups[0]["lr"])
            resume_amp_scale = float(scaler.get_scale())
            resume_optimizer_sha256 = _state_sha256(engine.state_dict())
            resume_model_sha256 = _parameter_sha256(unwrapped)
        task_id = f"{args.seed}-{args.route}"
        records: list[dict[str, object]] = []
        epoch_times: list[float] = []
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
                forward_start = time.perf_counter()
                batch = _to_device(batch, device)
                model_batch = workspace._apply_train_augmentation(batch)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(model_batch, training=True)
                forward_s = time.perf_counter() - forward_start
                if resume_oracle is not None:
                    resumed_facts = ResumeFacts(
                        next_batch_indices=batch_indices,
                        learning_rate=resume_learning_rate,
                        amp_scale=resume_amp_scale,
                        optimizer_state_sha256=resume_optimizer_sha256,
                        model_sha256=resume_model_sha256,
                        next_loss=float(loss.detach()),
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
                    _write_resume_oracle(
                        oracle_path,
                        ResumeFacts(
                            next_batch_indices=expected_indices,
                            learning_rate=expected_lr,
                            amp_scale=expected_scale,
                            optimizer_state_sha256=expected_optimizer_sha256,
                            model_sha256=expected_model_sha256,
                            next_loss=float(loss.detach()),
                        ),
                    )
                    pending_resume = None
                backward_start = time.perf_counter()
                scaler.scale(loss).backward()
                backward_total_s = time.perf_counter() - backward_start
                update = engine.step(scaler)
                if not bool(update["skipped"]):
                    scheduler.step()
                global_step += 1
                communication_s = float(update["communication_s"])
                backward_s = max(0.0, backward_total_s - communication_s)
                quality_start = time.perf_counter()
                unwrapped = model.module if hasattr(model, "module") else model
                rank_gap = _rank_gap(unwrapped, process_group)
                model_sha256 = _parameter_sha256(unwrapped)
                quality_s = time.perf_counter() - quality_start
                timing = StepTiming(
                    forward_s=float(forward_s),
                    backward_s=float(backward_s),
                    update_s=float(update["update_s"]),
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
                    loss=float(loss.detach()),
                    amp_scale=float(scaler.get_scale()),
                    learning_rate=float(workspace.optimizer.param_groups[0]["lr"]),
                    model_sha256=model_sha256,
                    rank_parameter_gap=rank_gap,
                    optimizer_step=engine.step_count,
                    finite=bool(torch.isfinite(loss)),
                )
                records.append(record)
                if rank == 0:
                    serialization_start = time.perf_counter()
                    with raw_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    record["timing"]["report_serialization_s"] += (
                        time.perf_counter() - serialization_start
                    )
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
                    pending_resume = (
                        checkpoint.with_suffix(".oracle.json"),
                        next_indices,
                        float(workspace.optimizer.param_groups[0]["lr"]),
                        float(scaler.get_scale()),
                        _state_sha256(engine.state_dict()),
                        model_sha256,
                    )
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
            summary = summarize_step_records(
                tuple(records),
                warmup_steps=args.warmup_steps,
                batch_size_per_rank=args.batch_size,
                world_size=world_size,
            )
            manifest = source_tree_manifest(args.psi_source)
            physical = tuple(
                int(value)
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip()
            )
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
                steps=global_step,
                warmup_steps=args.warmup_steps,
                steady_samples_per_second=float(summary["steady_samples_per_second"]),
                step_latency_p50_ms=float(summary["step_latency_p50_ms"]),
                step_latency_p95_ms=float(summary["step_latency_p95_ms"]),
                epoch_time_s=tuple(epoch_times),
                communication_time_s=float(summary["communication_time_s"]),
                qwd_time_s=float(summary["qwd_time_s"]),
                refresh_time_s=float(summary["refresh_time_s"]),
                communication_bytes=int(summary["communication_bytes"]),
                peak_memory_mib=float(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                ),
                gpu_telemetry=_gpu_telemetry(),
                loss_trajectory=tuple(summary["loss_trajectory"]),
                validation_loss=float(validation_loss),
                rank_gaps=tuple(summary["rank_gaps"]),
                decision_counts=summary["decision_counts"],
                failure_facts=(),
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
