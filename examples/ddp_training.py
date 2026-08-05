"""Reproducible native-DDP and CCDL end-to-end training comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.training.config import MODES, TrainingConfig
from examples.training.data import build_dataset
from examples.training.metrics import (
    CorrectnessMetrics,
    ExecutionMetrics,
    MemoryMetrics,
    TimingMetrics,
    TrainingResult,
)
from examples.training.model import build_mlp, count_parameters
from examples.training.overlap import CudaOverlapRecorder, OverlapMeasurement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native DDP with synchronous/asynchronous CCDL hooks."
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
    parser.add_argument("--no-error-feedback", action="store_false", dest="error_feedback")
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    values: dict[str, Any] = {}
    if args.config is not None:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
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
    if "mode" not in values:
        raise ValueError("mode must be provided by --mode or --config")
    return TrainingConfig(**values)


def run_training(config: TrainingConfig) -> dict[str, object] | None:
    import torch
    import torch.distributed as dist

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
        parameter_count = count_parameters(model)
        execution = _wrap_distributed_model(
            model,
            config=config,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        model = execution[0]
        execution_metrics = execution[1]
        overlap_recorder = execution[2]
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        loader = _build_loader(
            config,
            rank=rank,
            world_size=world_size,
            torch=torch,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        measured_latencies: list[float] = []
        iterator = iter(loader)
        for step in range(config.steps):
            features, targets = next(iterator)
            features = features.to(device=device, dtype=model_dtype, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            _synchronize(device, torch=torch)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.float(), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at rank={rank}, step={step}")
            measure_step = step >= config.warmup_steps
            if measure_step:
                overlap_recorder.begin_backward()
            loss.backward()
            if measure_step:
                overlap_recorder.end_backward()
            optimizer.step()
            _synchronize(device, torch=torch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            losses.append(float(loss.detach()))
            if step >= config.warmup_steps:
                measured_latencies.append(elapsed_ms)

        losses = _mean_rank_values(losses, device=device, world_size=world_size, torch=torch)
        measured_latencies = _max_rank_values(
            measured_latencies,
            device=device,
            world_size=world_size,
            torch=torch,
        )
        overlap, overlap_classification = overlap_recorder.collect()
        overlap = _mean_rank_overlap(
            overlap,
            device=device,
            world_size=world_size,
            torch=torch,
        )
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
        result = TrainingResult(
            mode=config.mode,
            world_size=world_size,
            global_batch_size=config.batch_size_per_rank * world_size,
            parameter_count=parameter_count,
            timing=TimingMetrics(
                measured_steps=config.measured_steps,
                elapsed_seconds=sum(measured_latencies) / 1000.0,
                step_latencies_ms=tuple(measured_latencies),
                overlap_efficiency=overlap.overlap_efficiency(),
                communication_ms=overlap.communication_ms,
                compute_ms=overlap.compute_ms,
                overlapped_ms=overlap.overlapped_ms,
                exposed_communication_ms=overlap.exposed_communication_ms,
                overlap_classification=overlap_classification,
            ),
            memory=MemoryMetrics(peak_allocated_bytes=peak_memory),
            losses=tuple(losses),
            correctness=correctness,
            execution=execution_metrics,
        )
        return result.to_dict() if rank == 0 else None
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _resolve_device(requested: str, *, local_rank: int, torch: Any) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _model_dtype(requested: str, *, device: Any, torch: Any) -> Any:
    if device.type == "cpu" and requested == "fp16":
        return torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[requested]


def _wrap_distributed_model(
    model: Any,
    *,
    config: TrainingConfig,
    device: Any,
    world_size: int,
    torch: Any,
) -> tuple[Any, ExecutionMetrics, CudaOverlapRecorder]:
    recorder = CudaOverlapRecorder(
        torch=torch,
        enabled=device.type == "cuda" and world_size > 1,
        asynchronous=config.mode == "ccdl_async",
    )
    if world_size == 1:
        return (
            model,
            ExecutionMetrics(
                requested_mode=config.mode,
                effective_strategy="single_rank",
                capability="not_exercised_world_size_1",
                fallback_reason=None,
            ),
            recorder,
        )
    kwargs = {"bucket_cap_mb": config.bucket_cap_mb}
    if device.type == "cuda":
        kwargs["device_ids"] = [device.index]
        kwargs["output_device"] = device.index
    distributed_model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
    if config.mode == "native_ddp":
        return (
            distributed_model,
            ExecutionMetrics(
                requested_mode=config.mode,
                effective_strategy="native_ddp",
                capability="torch_distributed",
                fallback_reason=None,
            ),
            recorder,
        )

    from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
    from ccdl_comm.config import CompressionConfig
    from ccdl_comm.cuda.loader import load_cuda_extension

    extension_status = load_cuda_extension()
    if device.type == "cuda" and not extension_status.available:
        raise RuntimeError(extension_status.reason or "CCDL CUDA extension unavailable")
    hook = create_ddp_comm_hook(
        CompressionConfig(
            bit=config.bit,
            group_size=config.group_size,
            error_feedback=config.error_feedback,
        ),
        dtype="auto",
        strategy="all_gather",
        reduce="mean",
        async_gather=config.mode == "ccdl_async",
        async_error_feedback=config.mode == "ccdl_async" and config.error_feedback,
        extension_status=extension_status,
    )
    distributed_model.register_comm_hook(state=None, hook=recorder.wrap_hook(hook))
    fallback = getattr(hook, "_ccdl_fallback_record", None)
    return (
        distributed_model,
        ExecutionMetrics(
            requested_mode=config.mode,
            effective_strategy=str(
                getattr(hook, "_ccdl_effective_strategy", "all_gather")
            ),
            capability=(
                "cuda_extension" if extension_status.available else "torch_fallback"
            ),
            fallback_reason=None if fallback is None else fallback.reason,
        ),
        recorder,
    )


def _build_loader(
    config: TrainingConfig,
    *,
    rank: int,
    world_size: int,
    torch: Any,
) -> Any:
    dataset = build_dataset(config, torch=torch, world_size=world_size)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size_per_rank,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def _synchronize(device: Any, *, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_rank_values(
    values: list[float],
    *,
    device: Any,
    world_size: int,
    torch: Any,
) -> list[float]:
    if world_size == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor)
    tensor /= world_size
    return tensor.cpu().tolist()


def _max_rank_values(
    values: list[float],
    *,
    device: Any,
    world_size: int,
    torch: Any,
) -> list[float]:
    if world_size == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.cpu().tolist()


def _mean_rank_overlap(
    measurement: OverlapMeasurement,
    *,
    device: Any,
    world_size: int,
    torch: Any,
) -> OverlapMeasurement:
    values = _mean_rank_values(
        [
            measurement.communication_ms,
            measurement.compute_ms,
            measurement.overlapped_ms,
            measurement.exposed_communication_ms,
        ],
        device=device,
        world_size=world_size,
        torch=torch,
    )
    return OverlapMeasurement(*values)


def _parameter_correctness(
    model: Any,
    *,
    device: Any,
    world_size: int,
    finite_loss: bool,
    torch: Any,
) -> CorrectnessMetrics:
    if world_size == 1:
        return CorrectnessMetrics(True, 0.0, finite_loss)
    maximum = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=0)
        maximum = torch.maximum(
            maximum,
            (parameter.detach() - reference).abs().float().max(),
        )
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    difference = float(maximum)
    return CorrectnessMetrics(difference == 0.0, difference, finite_loss)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    payload = run_training(config)
    if payload is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
