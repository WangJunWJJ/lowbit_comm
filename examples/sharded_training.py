"""Compare native DDP, full-gradient CCDL, and sharded SGD training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.ddp_training import run_training as run_ddp_training
from examples.ddp_training import (
    _build_loader,
    _max_rank_values,
    _mean_rank_values,
    _model_dtype,
    _parameter_correctness,
    _resolve_device,
    _synchronize,
)
from examples.training.config import TrainingConfig
from examples.training.metrics import (
    ExecutionMetrics,
    MemoryMetrics,
    TimingMetrics,
    TrainingResult,
)
from examples.training.model import build_mlp, count_parameters
from examples.training.sharded_metrics import (
    PHASE_NAMES,
    ShardedPhaseMetrics,
    augment_training_payload,
)
from examples.training.sharded_sgd import (
    TorchShardedSgdConsumer,
    compile_torch_shard_layout,
)


MODES = ("native_ddp", "ccdl_full_gradient", "ccdl_sharded_sgd")


@dataclass(frozen=True, slots=True)
class ShardedRunConfig:
    mode: str
    training: TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--no-error-feedback", action="store_false", dest="error_feedback")
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> ShardedRunConfig:
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
    training_mode = "ccdl_sync" if mode == "ccdl_full_gradient" else "native_ddp"
    return ShardedRunConfig(
        mode=mode,
        training=TrainingConfig(mode=training_mode, **values),
    )


def run(config: ShardedRunConfig) -> dict[str, object] | None:
    if config.mode == "ccdl_sharded_sgd":
        return _run_sharded_training(config.training)
    mapped_mode = "native_ddp" if config.mode == "native_ddp" else "ccdl_sync"
    payload = run_ddp_training(replace(config.training, mode=mapped_mode))
    if payload is None:
        return None
    phases = ShardedPhaseMetrics(
        measured_steps=config.training.measured_steps,
        samples_ms={
            name: (0.0,) * config.training.measured_steps for name in PHASE_NAMES
        },
    )
    return augment_training_payload(
        payload,
        mode=config.mode,
        phases=phases,
        phases_measured=False,
        initial_pointers={},
        final_pointers={},
    )


def _run_sharded_training(config: TrainingConfig) -> dict[str, object] | None:
    import torch
    import torch.distributed as dist

    from ccdl_comm.config import CompressionConfig
    from ccdl_comm.cuda.loader import load_cuda_extension
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
    from ccdl_comm.shard import ReducedShard

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(config.device, local_rank=local_rank, torch=torch)
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        initialized_here = True
    try:
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(config.seed)
        model_dtype = _model_dtype(config.dtype, device=device, torch=torch)
        model = build_mlp(config, torch=torch).to(device=device, dtype=model_dtype)
        active_parameters = tuple(model.parameters())
        parameter_count = count_parameters(model)
        layout = compile_torch_shard_layout(
            active_parameters,
            rank=rank,
            world_size=world_size,
        )

        def all_gather_into_tensor(output: Any, local: Any) -> Any:
            if world_size == 1:
                output.copy_(local)
                return None
            return dist.all_gather_into_tensor(output, local)

        consumer = TorchShardedSgdConsumer(
            active_parameters,
            layout=layout,
            learning_rate=config.learning_rate,
            all_gather_into_tensor=all_gather_into_tensor,
            torch=torch,
        )
        initial_pointers = consumer.buffer_pointers()
        compiled_plan = None
        extension_status = None
        if device.type == "cuda":
            extension_status = load_cuda_extension()
            if not extension_status.available:
                raise RuntimeError(
                    extension_status.reason or "CCDL CUDA extension unavailable"
                )
            compiled_plan = compile_cuda_shortcut(
                consumer.flatten_gradients(),
                collective="reduce_scatter",
                strategy="compressed",
                output_layout="shard",
                config=CompressionConfig(
                    bit=config.bit,
                    group_size=config.group_size,
                    error_feedback=config.error_feedback,
                ),
                async_op=False,
                dtype=config.dtype,
                extension_status=extension_status,
            )
            if compiled_plan.execution_info.fallback_used:
                raise RuntimeError(
                    compiled_plan.execution_info.fallback_reason
                    or "compressed reduce-scatter unexpectedly used fallback"
                )

        criterion = torch.nn.CrossEntropyLoss()
        loader = _build_loader(
            config,
            rank=rank,
            world_size=world_size,
            torch=torch,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        phase_timer = _PhaseTimer(torch=torch, device=device)
        losses: list[float] = []
        measured_latencies: list[float] = []
        iterator = iter(loader)
        for step in range(config.steps):
            features, targets = next(iterator)
            features = features.to(device=device, dtype=model_dtype, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            _synchronize(device, torch=torch)
            started = time.perf_counter()
            model.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.float(), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at rank={rank}, step={step}")
            measured = step >= config.warmup_steps

            def backward_and_flatten() -> Any:
                loss.backward()
                return consumer.flatten_gradients()

            flat_gradients = phase_timer.measure(
                "backward_and_flatten",
                backward_and_flatten,
                measured=measured,
            )

            def reduce_scatter() -> ReducedShard:
                if compiled_plan is not None:
                    return compiled_plan.run(
                        flat_gradients,
                        out=consumer.reduced_output(),
                    ).wait()
                consumer.reduced_output().copy_(flat_gradients)
                return ReducedShard(
                    shard=consumer.reduced_output(),
                    shard_index=rank,
                    shard_numel=layout.shard_numel,
                    original_shape=(layout.original_numel,),
                    original_numel=layout.original_numel,
                    padded_numel=layout.padded_numel,
                    world_size=world_size,
                    reduce="mean",
                    dtype=layout.dtype,
                    transport="single_rank",
                    metadata={"output_ownership": "caller"},
                )

            reduced = phase_timer.measure(
                "compressed_reduce_scatter",
                reduce_scatter,
                measured=measured,
            )
            phase_timer.measure(
                "local_shard_update",
                lambda: consumer.update_local(reduced),
                measured=measured,
            )
            phase_timer.measure(
                "parameter_all_gather",
                consumer.gather_parameters,
                measured=measured,
            )
            phase_timer.measure(
                "parameter_writeback",
                consumer.writeback_parameters,
                measured=measured,
            )
            _synchronize(device, torch=torch)
            phase_timer.complete_step()
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
        phase_samples = {
            name: tuple(
                _max_rank_values(
                    list(phase_timer.samples[name]),
                    device=device,
                    world_size=world_size,
                    torch=torch,
                )
            )
            for name in PHASE_NAMES
        }
        correctness = _parameter_correctness(
            model,
            device=device,
            world_size=world_size,
            finite_loss=all(value == value for value in losses),
            torch=torch,
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
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
            mode="ccdl_sharded_sgd",
            world_size=world_size,
            global_batch_size=config.batch_size_per_rank * world_size,
            parameter_count=parameter_count,
            workload=config.comparison_workload(),
            timing=TimingMetrics(
                measured_steps=config.measured_steps,
                elapsed_seconds=sum(measured_latencies) / 1000.0,
                step_latencies_ms=tuple(measured_latencies),
                overlap_classification="not_measured",
            ),
            memory=MemoryMetrics(peak_allocated_bytes=peak_memory),
            losses=tuple(losses),
            correctness=correctness,
            execution=ExecutionMetrics(
                requested_mode="ccdl_sharded_sgd",
                effective_strategy=(
                    "single_rank" if execution_info is None else execution_info.executed_strategy
                ),
                capability=(
                    "not_exercised_world_size_1"
                    if extension_status is None
                    else "cuda_extension"
                ),
                fallback_reason=(
                    None if execution_info is None else execution_info.fallback_reason
                ),
            ),
        ).to_dict()
        if execution_info is not None:
            result["execution"]["fast_path"] = execution_info.fast_path
            result["execution"]["output_layout"] = "shard"
        return (
            augment_training_payload(
                result,
                mode="ccdl_sharded_sgd",
                phases=ShardedPhaseMetrics(
                    measured_steps=config.measured_steps,
                    samples_ms=phase_samples,
                ),
                phases_measured=True,
                initial_pointers=initial_pointers,
                final_pointers=consumer.buffer_pointers(),
            )
            if rank == 0
            else None
        )
    finally:
        if initialized_here:
            dist.destroy_process_group()


class _PhaseTimer:
    def __init__(self, *, torch: Any, device: Any) -> None:
        self._torch = torch
        self._device = device
        self.samples = {name: [] for name in PHASE_NAMES}
        self._pending: list[tuple[str, Any, Any]] = []

    def measure(self, name: str, operation: Any, *, measured: bool) -> Any:
        if not measured:
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


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    payload = run(config)
    if payload is not None:
        config.training.output.parent.mkdir(parents=True, exist_ok=True)
        config.training.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
