"""Shared flat parameter storage for compressed sharded training examples."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import lcm
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from ccdl_comm.shard_layout import FlatShardLayout
from examples.training.sharded_sgd import compile_torch_shard_layout


MODES = (
    "native_ddp",
    "full_fused",
    "sharded_fp",
    "sharded_compressed",
    "sharded_qwd",
)
PIPELINE_STAGE_NAMES = (
    "backward_flatten",
    "compressed_reduce_scatter",
    "local_update",
    "parameter_quantize_gather",
    "parameter_restore_writeback",
)
QWD_PIPELINE_STAGE_NAMES = (
    "backward_flatten",
    "compressed_reduce_scatter",
    "local_update",
    "parameter_delta_quantize",
    "parameter_all_gather",
    "parameter_add_writeback",
    "fp_refresh",
)


@dataclass(frozen=True, slots=True)
class CompressedShardedRunConfig:
    mode: str
    training: Any


class TorchFlatParameterStorage:
    """Own padded flat storage directly viewed by all model parameters."""

    def __init__(
        self,
        *,
        parameters: tuple[Any, ...],
        padded_flat: Any,
        layout: FlatShardLayout,
    ) -> None:
        self._parameters = parameters
        self._padded_flat = padded_flat
        self._layout = layout

    @classmethod
    def from_parameters(
        cls,
        parameters: Iterable[Any],
        *,
        rank: int,
        world_size: int,
        group_size: int = 64,
    ) -> "TorchFlatParameterStorage":
        """Copy and atomically rebind homogeneous parameters to aligned storage."""

        _require_nonnegative_integer(rank, "rank")
        _require_positive_integer(world_size, "world_size")
        _require_positive_integer(group_size, "group_size")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")

        active = tuple(parameters)
        base_layout = compile_torch_shard_layout(
            active,
            rank=rank,
            world_size=world_size,
        )
        _validate_homogeneous_parameters(active)

        shard_alignment = lcm(group_size, 512)
        shard_numel = (
            _ceil_div(
                base_layout.original_numel,
                world_size * shard_alignment,
            )
            * shard_alignment
        )
        layout = FlatShardLayout(
            original_numel=base_layout.original_numel,
            padded_numel=shard_numel * world_size,
            shard_numel=shard_numel,
            world_size=world_size,
            shard_index=rank,
            parameters=base_layout.parameters,
        )
        padded_flat = active[0].new_zeros((layout.padded_numel,))

        for parameter, parameter_slice in zip(
            active,
            layout.parameters,
            strict=True,
        ):
            padded_flat.narrow(
                0,
                parameter_slice.offset,
                parameter_slice.numel,
            ).copy_(parameter.detach().reshape(-1))

        original_data = tuple(parameter.data for parameter in active)
        try:
            for parameter, parameter_slice in zip(
                active,
                layout.parameters,
                strict=True,
            ):
                parameter.data = padded_flat.narrow(
                    0,
                    parameter_slice.offset,
                    parameter_slice.numel,
                ).view(parameter_slice.shape)
        except Exception:
            for parameter, previous in zip(active, original_data, strict=True):
                parameter.data = previous
            raise

        return cls(
            parameters=active,
            padded_flat=padded_flat,
            layout=layout,
        )

    @property
    def layout(self) -> FlatShardLayout:
        return self._layout

    @property
    def padded_flat(self) -> Any:
        return self._padded_flat

    @property
    def original_numel(self) -> int:
        return self._layout.original_numel

    @property
    def local_shard(self) -> Any:
        return self._padded_flat.narrow(
            0,
            self._layout.shard_offset,
            self._layout.shard_numel,
        )

    @property
    def parameters(self) -> tuple[Any, ...]:
        return self._parameters

    def flatten_gradients(self, *, out: Any | None = None) -> Any:
        """Copy parameter gradients into caller-owned padded flat storage."""

        target = self._padded_flat.new_zeros((self._layout.padded_numel,)) if out is None else out
        if int(target.numel()) != self._layout.padded_numel:
            raise ValueError("gradient output numel must equal padded_numel")
        target.zero_()
        for parameter, parameter_slice in zip(
            self._parameters,
            self._layout.parameters,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is None:
                continue
            target.narrow(
                0,
                parameter_slice.offset,
                parameter_slice.numel,
            ).copy_(gradient.detach().reshape(-1))
        return target

    def buffer_pointers(self) -> dict[str, int]:
        return {
            "padded_flat": int(self._padded_flat.data_ptr()),
            "local_shard": int(self.local_shard.data_ptr()),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native, fused, and sharded CCDL parameter pipelines."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=MODES)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("--synthetic", action="store_true", default=None)
    dataset.add_argument("--data-root", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--batch-size-per-rank", type=int)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--bit", type=int, choices=(4, 8))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--bucket-cap-mb", type=int)
    parser.add_argument("--error-feedback", action="store_true", default=None)
    parser.add_argument(
        "--no-error-feedback",
        action="store_false",
        dest="error_feedback",
    )
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> CompressedShardedRunConfig:
    from examples.training.config import TrainingConfig

    values: dict[str, Any] = {}
    if args.config is not None:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("config JSON must contain an object")
        values.update(payload)
    for name in (
        "mode",
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
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    mode = values.pop("mode", None)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    training_mode = "ccdl_sync" if mode == "full_fused" else "native_ddp"
    return CompressedShardedRunConfig(
        mode=mode,
        training=TrainingConfig(mode=training_mode, **values),
    )


def run_fake_step(*, mode: str) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    result = {
        "mode": mode,
        "stage_ms": {
            name: 0.0
            for name in (
                QWD_PIPELINE_STAGE_NAMES
                if mode == "sharded_qwd"
                else PIPELINE_STAGE_NAMES
            )
        },
        "selected_fast_path": (
            "compressed_parameter_restore"
            if mode == "sharded_compressed"
            else "fused_int8_qwd" if mode == "sharded_qwd" else mode
        ),
        "fallback_reason": None,
    }
    if mode == "sharded_qwd":
        result["parameter_communication"] = {
            "algorithm": "qwd",
            "bit": 8,
            "warmup_steps": 0,
            "refresh_interval": 512,
            "relative_error_threshold": 1.0e-2,
            "decision_counts": {"qwd": 1, "fp_refresh": 0},
            "sampled_relative_errors": [],
        }
    return result


def run_training(config: CompressedShardedRunConfig) -> dict[str, object] | None:
    """Run one comparable parameter-communication benchmark mode."""

    if config.mode == "sharded_compressed":
        return _run_compressed_sharded(config.training)
    if config.mode == "sharded_qwd":
        return _run_qwd_sharded(config.training)
    return _run_existing_baseline(config)


def _run_existing_baseline(config: CompressedShardedRunConfig) -> dict[str, object] | None:
    from examples.sharded_training import ShardedRunConfig, run

    mapped = {
        "native_ddp": "native_ddp",
        "full_fused": "ccdl_full_gradient",
        "sharded_fp": "ccdl_sharded_sgd",
    }[config.mode]
    training = replace(
        config.training,
        mode="ccdl_sync" if mapped == "ccdl_full_gradient" else "native_ddp",
    )
    payload = run(ShardedRunConfig(mode=mapped, training=training))
    if payload is None:
        return None
    result = dict(payload)
    result["mode"] = config.mode
    execution = dict(result.get("execution", {}))
    execution["requested_mode"] = config.mode
    result["execution"] = execution
    phase = result.get("phase_timing_ms", {})
    result["stage_ms"] = {
        "backward_flatten": float(phase.get("backward_and_flatten", 0.0)),
        "compressed_reduce_scatter": float(
            phase.get("compressed_reduce_scatter", 0.0)
        ),
        "local_update": float(phase.get("local_shard_update", 0.0)),
        "parameter_quantize_gather": float(
            phase.get("parameter_all_gather", 0.0)
        ),
        "parameter_restore_writeback": float(
            phase.get("parameter_writeback", 0.0)
        ),
    }
    result["selected_fast_path"] = execution.get(
        "fast_path",
        execution.get("effective_strategy", mapped),
    )
    result["fallback_reason"] = execution.get("fallback_reason")
    result["workspace_pointers"] = result.get("buffer_reuse", {})
    _require_correct_result(result)
    return result


def _run_compressed_sharded(training: Any) -> dict[str, object] | None:
    import torch
    import torch.distributed as dist

    from ccdl_comm.communication import (
        ShardedStepPipeline,
        TorchCompressedParameterRestore,
    )
    from ccdl_comm.config import CompressionConfig
    from ccdl_comm.cuda.loader import load_cuda_extension
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
    from ccdl_comm.optim import SgdShardUpdateRule, ShardedOptimizerConsumer
    from examples.ddp_training import (
        _build_loader,
        _max_rank_values,
        _mean_rank_values,
        _model_dtype,
        _parameter_correctness,
        _resolve_device,
        _synchronize,
    )
    from examples.training.metrics import (
        ExecutionMetrics,
        MemoryMetrics,
        TimingMetrics,
        TrainingResult,
    )
    from examples.training.model import build_mlp, count_parameters
    from examples.training.sharded_sgd import exact_mean_reduce_scatter

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(training.device, local_rank=local_rank, torch=torch)
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        initialized_here = True
    try:
        torch.manual_seed(training.seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(training.seed)
        model_dtype = _model_dtype(training.dtype, device=device, torch=torch)
        model = build_mlp(training, torch=torch).to(device=device, dtype=model_dtype)
        storage = TorchFlatParameterStorage.from_parameters(
            model.parameters(),
            rank=rank,
            world_size=world_size,
            group_size=training.group_size,
        )
        flat_gradients = storage.padded_flat.new_zeros((storage.layout.padded_numel,))
        reduced_output = storage.local_shard.new_empty((storage.layout.shard_numel,))
        compression = CompressionConfig(
            bit=training.bit,
            group_size=training.group_size,
            error_feedback=training.error_feedback,
            compact=True,
            allow_experimental=training.bit != 8,
        )
        extension_status = load_cuda_extension()
        compiled_plan = None
        if device.type == "cuda" and world_size > 1:
            if not extension_status.available:
                raise RuntimeError(
                    extension_status.reason or "CCDL CUDA extension unavailable"
                )
            compiled_plan = compile_cuda_shortcut(
                flat_gradients,
                collective="reduce_scatter",
                strategy="compressed",
                output_layout="shard",
                config=compression,
                async_op=False,
                dtype=training.dtype,
                extension_status=extension_status,
            )
            if compiled_plan.execution_info.fallback_used:
                raise RuntimeError(
                    compiled_plan.execution_info.fallback_reason
                    or "compressed reduce-scatter unexpectedly used fallback"
                )

        distributed_facade = dist if world_size > 1 else _SingleRankDistributed()
        restore = TorchCompressedParameterRestore(
            config=compression,
            dtype=training.dtype,
            import_module=lambda name: (
                distributed_facade
                if name == "torch.distributed"
                else __import__(name)
            ),
            extension_status=extension_status,
        )
        optimizer_consumer = ShardedOptimizerConsumer(
            layout=storage.layout,
            parameter_shard=storage.local_shard,
            update_rule=SgdShardUpdateRule(training.learning_rate),
        )
        stage_timer = _StageTimer(torch=torch, device=device)
        timed_consumer = _TimedConsumer(optimizer_consumer, stage_timer)
        timed_restore = _TimedRestore(restore, stage_timer)
        pipeline = ShardedStepPipeline(
            consumer_for_bucket=lambda bucket_id: timed_consumer,
            restore_for_bucket=lambda bucket_id: timed_restore,
            max_inflight=2,
        )
        initial_pointers: dict[str, object] | None = None
        criterion = torch.nn.CrossEntropyLoss()
        loader = _build_loader(
            training,
            rank=rank,
            world_size=world_size,
            torch=torch,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        measured_latencies: list[float] = []
        iterator = iter(loader)
        for step_index in range(training.steps):
            features, targets = next(iterator)
            features = features.to(device=device, dtype=model_dtype, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            _synchronize(device, torch=torch)
            started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.float(), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at rank={rank}, step={step_index}"
                )
            measured = step_index >= training.warmup_steps
            stage_timer.measured = measured

            def backward_flatten() -> Any:
                loss.backward()
                return storage.flatten_gradients(out=flat_gradients)

            gradients = stage_timer.measure("backward_flatten", backward_flatten)

            def reduce_scatter() -> Any:
                if compiled_plan is not None:
                    return compiled_plan.run(gradients, out=reduced_output).wait()
                return exact_mean_reduce_scatter(
                    gradients,
                    out=reduced_output,
                    layout=storage.layout,
                    reduce_scatter_tensor=distributed_facade.reduce_scatter_tensor,
                )

            reduced = stage_timer.measure(
                "compressed_reduce_scatter",
                reduce_scatter,
            )
            reduced = storage.layout.bind_reduced_shard(reduced)
            pipeline.consume_bucket(
                "model",
                reduced,
                parameter_view=storage.padded_flat,
                step=step_index + 1,
            )
            stage_timer.measure("parameter_restore_writeback", pipeline.finish_step)
            _synchronize(device, torch=torch)
            stage_timer.complete_step()
            if initial_pointers is None:
                initial_pointers = _workspace_pointers(
                    storage,
                    flat_gradients,
                    reduced_output,
                    restore,
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            losses.append(float(loss.detach()))
            if measured:
                measured_latencies.append(elapsed_ms)

        losses = _mean_rank_values(
            losses,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        measured_latencies = _max_rank_values(
            measured_latencies,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        stage_samples = {
            name: _max_rank_values(
                list(stage_timer.samples[name]),
                device=device,
                world_size=world_size,
                torch=torch,
            )
            for name in PIPELINE_STAGE_NAMES
        }
        correctness = _parameter_correctness(
            model,
            device=device,
            world_size=world_size,
            finite_loss=all(value == value for value in losses),
            torch=torch,
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        peak_memory = int(
            _max_rank_values(
                [float(peak_memory)],
                device=device,
                world_size=world_size,
                torch=torch,
            )[0]
        )
        execution_info = None if compiled_plan is None else compiled_plan.execution_info
        result = TrainingResult(
            mode="sharded_compressed",
            world_size=world_size,
            global_batch_size=training.batch_size_per_rank * world_size,
            parameter_count=count_parameters(model),
            workload=training.comparison_workload(),
            timing=TimingMetrics(
                measured_steps=training.measured_steps,
                elapsed_seconds=sum(measured_latencies) / 1000.0,
                step_latencies_ms=tuple(measured_latencies),
                overlap_classification="bounded_sharded_restore",
            ),
            memory=MemoryMetrics(peak_allocated_bytes=peak_memory),
            losses=tuple(losses),
            correctness=correctness,
            execution=ExecutionMetrics(
                requested_mode="sharded_compressed",
                effective_strategy="compressed_reduce_scatter_parameter_restore",
                capability=(
                    "cuda_extension"
                    if extension_status.available
                    else "fp_parameter_restore_fallback"
                ),
                fallback_reason=(
                    restore.last_fallback_reason
                    or (
                        None
                        if execution_info is None
                        else execution_info.fallback_reason
                    )
                ),
            ),
        ).to_dict()
        result["stage_ms"] = {
            name: fmean(stage_samples[name]) for name in PIPELINE_STAGE_NAMES
        }
        result["selected_fast_path"] = restore.last_fast_path
        result["fallback_reason"] = restore.last_fallback_reason
        result["max_rank_parameter_difference"] = correctness.max_parameter_difference
        result["workspace_pointers"] = {
            "initial": initial_pointers or {},
            "final": _workspace_pointers(
                storage,
                flat_gradients,
                reduced_output,
                restore,
            ),
        }
        result["workspace_pointers"]["stable"] = (
            result["workspace_pointers"]["initial"]
            == result["workspace_pointers"]["final"]
        )
        _require_correct_result(result)
        return result if rank == 0 else None
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _run_qwd_sharded(training: Any) -> dict[str, object] | None:
    import torch
    import torch.distributed as dist

    from ccdl_comm.communication import (
        SafeInt8QWDPolicy,
        TorchQuantizedParameterDeltaRestore,
    )
    from ccdl_comm.config import CompressionConfig
    from ccdl_comm.cuda.loader import load_cuda_extension
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
    from ccdl_comm.quantization.codec import (
        inplace_dequantize_gathered_add,
        quantize_parameter_delta,
        quantize_tensor,
    )
    from examples.ddp_training import (
        _build_loader,
        _max_rank_values,
        _mean_rank_values,
        _model_dtype,
        _parameter_correctness,
        _resolve_device,
        _synchronize,
    )
    from examples.training.metrics import (
        ExecutionMetrics,
        MemoryMetrics,
        TimingMetrics,
        TrainingResult,
    )
    from examples.training.model import build_mlp, count_parameters
    from examples.training.sharded_sgd import exact_mean_reduce_scatter
    from examples.training.torch_sharded_adamw import TorchShardedAdamWStep

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(training.device, local_rank=local_rank, torch=torch)
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        initialized_here = True
    try:
        torch.manual_seed(training.seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(training.seed)
        model_dtype = _model_dtype(training.dtype, device=device, torch=torch)
        model = build_mlp(training, torch=torch).to(
            device=device,
            dtype=model_dtype,
        )
        compression = CompressionConfig(
            bit=training.bit,
            group_size=training.group_size,
            error_feedback=True,
            compact=True,
            allow_experimental=training.bit != 8,
        )
        if compression.bit != 8:
            raise ValueError("sharded_qwd requires bit=8")
        extension_status = load_cuda_extension()
        if device.type != "cuda" or not extension_status.available:
            raise RuntimeError(
                extension_status.reason
                or "sharded_qwd requires the CCDL CUDA extension"
            )
        policy = SafeInt8QWDPolicy(
            warmup_steps=training.warmup_steps,
            refresh_interval=512,
            relative_error_threshold=1.0e-2,
            error_check_interval=128,
        )
        stage_timer = _StageTimer(torch=torch, device=device)
        distributed_facade = dist if world_size > 1 else _SingleRankDistributed()
        timed_distributed = _QWDTimedDistributed(distributed_facade, stage_timer)

        def timed_quantize(tensor, config, *, output):
            return stage_timer.measure(
                "parameter_delta_quantize",
                lambda: quantize_tensor(
                    tensor,
                    config,
                    output=output,
                    extension_status=extension_status,
                ),
            )

        def timed_quantize_difference(
            master,
            model_shard,
            config,
            *,
            output,
            valid_numel,
        ):
            return stage_timer.measure(
                "parameter_delta_quantize",
                lambda: quantize_parameter_delta(
                    master,
                    model_shard,
                    config,
                    output=output,
                    valid_numel=valid_numel,
                    extension_status=extension_status,
                ),
            )

        def timed_dequantize_add(
            gathered,
            out,
            decoded,
            config,
            **kwargs,
        ):
            del decoded
            arguments = dict(kwargs)
            arguments.pop("dtype")
            return stage_timer.measure(
                "parameter_add_writeback",
                lambda: inplace_dequantize_gathered_add(
                    gathered,
                    out,
                    config,
                    extension_status=extension_status,
                    **arguments,
                ),
            )

        def timed_overwrite(out, gathered):
            return stage_timer.measure(
                "fp_refresh",
                lambda: out.copy_(gathered),
            )

        restore = TorchQuantizedParameterDeltaRestore(
            config=compression,
            model_dtype=training.dtype,
            import_module=lambda name: (
                timed_distributed
                if name == "torch.distributed"
                else __import__(name)
            ),
            quantize=timed_quantize,
            quantize_difference=timed_quantize_difference,
            dequantize_add=timed_dequantize_add,
            overwrite=timed_overwrite,
            extension_status=extension_status,
        )
        compiled_plan = None

        def reduce_scatter(flattened, *, out, layout):
            nonlocal compiled_plan
            if world_size == 1:
                return exact_mean_reduce_scatter(
                    flattened,
                    out=out,
                    layout=layout,
                    reduce_scatter_tensor=distributed_facade.reduce_scatter_tensor,
                )
            if compiled_plan is None:
                compiled_plan = compile_cuda_shortcut(
                    flattened,
                    collective="reduce_scatter",
                    strategy="compressed",
                    output_layout="shard",
                    config=compression,
                    async_op=False,
                    dtype=training.dtype,
                    extension_status=extension_status,
                )
                if compiled_plan.execution_info.fallback_used:
                    raise RuntimeError(
                        compiled_plan.execution_info.fallback_reason
                        or "compressed reduce-scatter unexpectedly used fallback"
                    )
            return compiled_plan.run(flattened, out=out).wait()

        def global_error_ratio(residual_norm_sq, delta_norm_sq) -> float:
            if world_size == 1:
                return float(
                    (
                        residual_norm_sq.float()
                        / delta_norm_sq.float().clamp_min(1.0e-24)
                    ).sqrt()
                )
            norms = torch.stack(
                (residual_norm_sq.float(), delta_norm_sq.float())
            )
            dist.all_reduce(norms, op=dist.ReduceOp.SUM)
            return float((norms[0] / norms[1].clamp_min(1.0e-24)).sqrt())

        adapter = TorchShardedAdamWStep.from_parameters(
            model.parameters(),
            rank=rank,
            world_size=world_size,
            group_size=training.group_size,
            learning_rate=training.learning_rate,
            reduce_scatter=reduce_scatter,
            restore=restore,
            policy=policy,
            global_error_ratio=global_error_ratio,
            stage_measure=stage_timer.measure,
        )
        criterion = torch.nn.CrossEntropyLoss()
        loader = _build_loader(
            training,
            rank=rank,
            world_size=world_size,
            torch=torch,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        measured_latencies: list[float] = []
        decision_counts = {"qwd": 0, "fp_refresh": 0}
        relative_errors: list[float] = []
        measured_fast_paths: set[str] = set()
        fallback_reasons: set[str] = set()
        initial_pointers: dict[str, object] | None = None
        iterator = iter(loader)
        for step_index in range(training.steps):
            features, targets = next(iterator)
            features = features.to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )
            targets = targets.to(device=device, non_blocking=True)
            _synchronize(device, torch=torch)
            started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.float(), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at rank={rank}, step={step_index}"
                )
            loss.backward()
            measured = step_index >= training.warmup_steps
            stage_timer.measured = measured
            metrics = adapter.step(step=step_index + 1)
            _synchronize(device, torch=torch)
            stage_timer.complete_step()
            losses.append(float(loss.detach()))
            if measured:
                measured_latencies.append(
                    (time.perf_counter() - started) * 1000.0
                )
                decision_counts[metrics.parameter_communication_mode] += 1
                if (
                    metrics.relative_error is not None
                    and (step_index + 1) % policy.error_check_interval == 0
                ):
                    relative_errors.append(metrics.relative_error)
                if restore.last_fast_path is not None:
                    measured_fast_paths.add(restore.last_fast_path)
                if restore.last_fallback_reason is not None:
                    fallback_reasons.add(restore.last_fallback_reason)
            if initial_pointers is None:
                initial_pointers = {
                    "adapter": adapter.workspace_pointers(),
                    "restore": restore.workspace_pointers(),
                }

        losses = _mean_rank_values(
            losses,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        measured_latencies = _max_rank_values(
            measured_latencies,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        stage_samples = {
            name: _max_rank_values(
                list(stage_timer.samples[name]),
                device=device,
                world_size=world_size,
                torch=torch,
            )
            for name in QWD_PIPELINE_STAGE_NAMES
        }
        correctness = _parameter_correctness(
            model,
            device=device,
            world_size=world_size,
            finite_loss=all(value == value for value in losses),
            torch=torch,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        peak_memory = int(
            _max_rank_values(
                [float(peak_memory)],
                device=device,
                world_size=world_size,
                torch=torch,
            )[0]
        )
        result = TrainingResult(
            mode="sharded_qwd",
            world_size=world_size,
            global_batch_size=training.batch_size_per_rank * world_size,
            parameter_count=count_parameters(model),
            workload=training.comparison_workload(),
            timing=TimingMetrics(
                measured_steps=training.measured_steps,
                elapsed_seconds=sum(measured_latencies) / 1000.0,
                step_latencies_ms=tuple(measured_latencies),
                overlap_classification="qwd_fused_parameter_restore",
            ),
            memory=MemoryMetrics(peak_allocated_bytes=peak_memory),
            losses=tuple(losses),
            correctness=correctness,
            execution=ExecutionMetrics(
                requested_mode="sharded_qwd",
                effective_strategy="compressed_reduce_scatter_qwd_all_gather",
                capability="cuda_extension",
                fallback_reason=(
                    None if not fallback_reasons else "; ".join(fallback_reasons)
                ),
            ),
        ).to_dict()
        final_pointers = {
            "adapter": adapter.workspace_pointers(),
            "restore": restore.workspace_pointers(),
        }
        result["stage_ms"] = {
            name: fmean(values) if values else 0.0
            for name, values in stage_samples.items()
        }
        result["selected_fast_path"] = (
            "fused_int8_qwd"
            if "fused_int8_qwd" in measured_fast_paths
            else restore.last_fast_path
        )
        result["fallback_reason"] = (
            None if not fallback_reasons else "; ".join(fallback_reasons)
        )
        result["max_rank_parameter_difference"] = (
            correctness.max_parameter_difference
        )
        result["workspace_pointers"] = {
            "initial": initial_pointers or {},
            "final": final_pointers,
            "stable": (initial_pointers or {}) == final_pointers,
        }
        result["parameter_communication"] = {
            "algorithm": "qwd",
            "bit": compression.bit,
            "warmup_steps": policy.warmup_steps,
            "refresh_interval": policy.refresh_interval,
            "relative_error_threshold": policy.relative_error_threshold,
            "decision_counts": decision_counts,
            "sampled_relative_errors": relative_errors,
        }
        _require_correct_result(result)
        return result if rank == 0 else None
    finally:
        if initialized_here:
            dist.destroy_process_group()


class _SingleRankDistributed:
    def get_world_size(self) -> int:
        return 1

    @staticmethod
    def all_gather_into_tensor(output: Any, value: Any, *, async_op: bool = False) -> None:
        del async_op
        output.copy_(value)

    @staticmethod
    def reduce_scatter_tensor(output: Any, value: Any) -> None:
        output.copy_(value)


class _QWDTimedDistributed:
    def __init__(self, distributed: Any, timer: "_StageTimer") -> None:
        self._distributed = distributed
        self._timer = timer

    def get_world_size(self) -> int:
        return int(self._distributed.get_world_size())

    def all_gather_into_tensor(
        self,
        output: Any,
        value: Any,
        *,
        async_op: bool = False,
    ) -> Any:
        def operation() -> Any:
            return self._distributed.all_gather_into_tensor(
                output,
                value,
                async_op=async_op,
            )
        if str(getattr(value, "dtype", "")) == "torch.uint8":
            return self._timer.measure("parameter_all_gather", operation)
        return operation()


class _TimedConsumer:
    def __init__(self, consumer: Any, timer: "_StageTimer") -> None:
        self._consumer = consumer
        self._timer = timer

    def consume(self, reduced: Any, *, step: int) -> Any:
        return self._timer.measure(
            "local_update",
            lambda: self._consumer.consume(reduced, step=step),
        )


class _TimedRestore:
    def __init__(self, restore: Any, timer: "_StageTimer") -> None:
        self._restore = restore
        self._timer = timer

    def restore(self, updated: Any, *, out: Any, async_op: bool) -> Any:
        return self._timer.measure(
            "parameter_quantize_gather",
            lambda: self._restore.restore(updated, out=out, async_op=async_op),
        )


class _StageTimer:
    def __init__(self, *, torch: Any, device: Any) -> None:
        self._torch = torch
        self._device = device
        self.measured = False
        self.samples: dict[str, list[float]] = {
            name: [] for name in QWD_PIPELINE_STAGE_NAMES
        }
        self._pending: list[tuple[str, Any, Any]] = []

    def measure(self, name: str, operation: Any) -> Any:
        if not self.measured:
            return operation()
        if self._device.type == "cuda":
            start = self._torch.cuda.Event(enable_timing=True)
            end = self._torch.cuda.Event(enable_timing=True)
            start.record()
            result = operation()
            end.record()
            self._pending.append((name, start, end))
            return result
        started = time.perf_counter()
        result = operation()
        self.samples[name].append((time.perf_counter() - started) * 1000.0)
        return result

    def complete_step(self) -> None:
        for name, start, end in self._pending:
            self.samples[name].append(float(start.elapsed_time(end)))
        self._pending.clear()


def _workspace_pointers(
    storage: TorchFlatParameterStorage,
    flat_gradients: Any,
    reduced_output: Any,
    restore: Any,
) -> dict[str, object]:
    return {
        **storage.buffer_pointers(),
        "flat_gradients": int(flat_gradients.data_ptr()),
        "reduced_output": int(reduced_output.data_ptr()),
        "restore": restore.workspace_pointers(),
    }


def _require_correct_result(result: dict[str, object]) -> None:
    correctness = result.get("correctness", {})
    if not correctness.get("finite_loss", False):
        raise FloatingPointError("training produced a non-finite loss")
    if float(correctness.get("max_parameter_difference", 0.0)) != 0.0:
        raise RuntimeError("rank parameter difference must be exactly zero")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    payload = run_training(config)
    if payload is not None:
        config.training.output.parent.mkdir(parents=True, exist_ok=True)
        config.training.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
    return 0


def _validate_homogeneous_parameters(parameters: tuple[Any, ...]) -> None:
    first = parameters[0]
    first_dtype = first.dtype
    first_device = first.device
    for parameter in parameters:
        if parameter.dtype != first_dtype:
            raise ValueError("all parameters must use the same dtype")
        if parameter.device != first_device:
            raise ValueError("all parameters must use the same device")
        if getattr(parameter, "layout", None) != getattr(first, "layout", None):
            raise ValueError("all parameters must use the same tensor layout")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _require_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_integer(value: object, name: str) -> None:
    _require_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be > 0")


__all__ = [
    "CompressedShardedRunConfig",
    "MODES",
    "PIPELINE_STAGE_NAMES",
    "QWD_PIPELINE_STAGE_NAMES",
    "TorchFlatParameterStorage",
    "build_parser",
    "config_from_args",
    "main",
    "run_fake_step",
    "run_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
