from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig


class SyntheticMLP(nn.Module):
    """Configurable dense model for DDP communication-pressure benchmarks."""

    def __init__(self, *, input_dim: int, width: int, depth: int, output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, width, bias=False), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width, bias=False), nn.GELU()])
        layers.append(nn.Linear(width, output_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "ccdl", "fsdp"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-dtype", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--strategy", choices=("all_gather", "all_reduce", "auto"), default="all_gather")
    parser.add_argument("--min-compress-numel", type=int, default=0)
    parser.add_argument("--error-feedback", choices=("true", "false"), default="true")
    parser.add_argument(
        "--error-feedback-policy",
        choices=("none", "always", "large_bucket_only", "warmup_then_enable", "periodic"),
        default="always",
    )
    parser.add_argument("--error-feedback-min-numel", type=int, default=0)
    parser.add_argument("--error-feedback-warmup-steps", type=int, default=0)
    parser.add_argument("--error-feedback-period", type=int, default=1)
    parser.add_argument("--async-gather", choices=("true", "false"), default="false")
    parser.add_argument("--async-error-feedback", choices=("true", "false"), default="false")
    return parser.parse_args()


def setup(seed: int) -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    return rank, dist.get_world_size(), torch.device("cuda", local_rank)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    local_rank = int(os.environ["LOCAL_RANK"])
    model_dtype = torch.float16 if args.model_dtype == "fp16" else torch.float32
    model = SyntheticMLP(
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=args.output_dim,
    ).to(device=device, dtype=model_dtype)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if args.mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel

        fsdp_model = FullyShardedDataParallel(model, device_id=device)
        fsdp_model._ccdl_parameter_count = parameter_count
        return fsdp_model

    ddp_model = DistributedDataParallel(model, device_ids=[local_rank], bucket_cap_mb=args.bucket_cap_mb)
    ddp_model._ccdl_parameter_count = parameter_count
    if args.mode == "ccdl":
        hook = create_ddp_comm_hook(
            CompressionConfig(
                bit=args.bit,
                group_size=args.group_size,
                error_feedback=(args.error_feedback == "true"),
                error_feedback_policy=args.error_feedback_policy,
                error_feedback_min_numel=args.error_feedback_min_numel,
                error_feedback_warmup_steps=args.error_feedback_warmup_steps,
                error_feedback_period=args.error_feedback_period,
            ),
            dtype="auto",
            strategy=args.strategy,
            reduce="mean",
            min_compress_numel=args.min_compress_numel,
            async_gather=(args.async_gather == "true"),
            async_error_feedback=(args.async_error_feedback == "true"),
        )
        ddp_model._ccdl_strategy_plan = getattr(hook, "_ccdl_strategy_plan", None)
        ddp_model._ccdl_effective_strategy = getattr(hook, "_ccdl_effective_strategy", args.strategy)
        ddp_model.register_comm_hook(
            state=None,
            hook=hook,
        )
    return ddp_model


def count_parameters(model: nn.Module) -> int:
    if hasattr(model, "_ccdl_parameter_count"):
        return int(model._ccdl_parameter_count)
    wrapped = getattr(model, "module", model)
    return sum(parameter.numel() for parameter in wrapped.parameters())


def train(args: argparse.Namespace) -> None:
    rank, world_size, device = setup(args.seed)
    model = build_model(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    measured_step_times: list[float] = []
    start = time.perf_counter()
    for step in range(args.steps):
        tensor_dtype = torch.float16 if args.model_dtype == "fp16" else torch.float32
        inputs = torch.randn(args.batch_size_per_rank, args.input_dim, device=device, dtype=tensor_dtype)
        targets = torch.randn(args.batch_size_per_rank, args.output_dim, device=device, dtype=tensor_dtype)
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at rank={rank} step={step}")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - step_start) * 1000
        losses.append(float(loss.detach()))
        if step >= args.warmup_steps:
            measured_step_times.append(step_ms)

    loss_total = torch.tensor([sum(losses), len(losses)], device=device, dtype=torch.float64)
    measured_total = torch.tensor(
        [sum(measured_step_times), len(measured_step_times)],
        device=device,
        dtype=torch.float64,
    )
    memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**20], device=device, dtype=torch.float64)
    dist.all_reduce(loss_total)
    dist.all_reduce(measured_total)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        params = count_parameters(model)
        avg_step_ms = float(measured_total[0] / measured_total[1])
        strategy_plan = getattr(model, "_ccdl_strategy_plan", None)
        selected_strategy = getattr(model, "_ccdl_effective_strategy", None) if args.mode == "ccdl" else None
        strategy_fallback_reason = getattr(strategy_plan, "reason", None) if args.mode == "ccdl" else None
        result = {
            "mode": args.mode,
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "global_batch_size": args.batch_size_per_rank * world_size,
            "input_dim": args.input_dim,
            "width": args.width,
            "depth": args.depth,
            "output_dim": args.output_dim,
            "parameter_count": params,
            "bucket_cap_mb": args.bucket_cap_mb,
            "model_dtype": args.model_dtype,
            "strategy": args.strategy if args.mode == "ccdl" else ("fsdp_default" if args.mode == "fsdp" else "ddp_default"),
            "selected_strategy": selected_strategy,
            "strategy_fallback_reason": strategy_fallback_reason,
            "bit": args.bit if args.mode == "ccdl" else None,
            "group_size": args.group_size if args.mode == "ccdl" else None,
            "min_compress_numel": args.min_compress_numel if args.mode == "ccdl" else None,
            "error_feedback": args.error_feedback if args.mode == "ccdl" else None,
            "error_feedback_policy": args.error_feedback_policy if args.mode == "ccdl" else None,
            "error_feedback_min_numel": args.error_feedback_min_numel if args.mode == "ccdl" else None,
            "error_feedback_warmup_steps": args.error_feedback_warmup_steps if args.mode == "ccdl" else None,
            "error_feedback_period": args.error_feedback_period if args.mode == "ccdl" else None,
            "async_gather": args.async_gather if args.mode == "ccdl" else None,
            "async_error_feedback": args.async_error_feedback if args.mode == "ccdl" else None,
            "train_loss": float(loss_total[0] / loss_total[1]),
            "avg_step_ms": avg_step_ms,
            "samples_per_s": float(args.batch_size_per_rank * world_size / (avg_step_ms / 1000)),
            "peak_memory_mb_max_rank": float(memory[0]),
            "elapsed_s_rank0": time.perf_counter() - start,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
