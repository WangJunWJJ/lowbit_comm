from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from ccdl.comm import qall_reduce
from ccdl.quantization import Quantizer


class SyntheticMLP(nn.Module):
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
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=2048)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--legacy-method", choices=("tree", "p2p", "gather", "ring"), default="tree")
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def setup(seed: int) -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    return rank, dist.get_world_size(), torch.device("cuda", local_rank)


def synchronize_gradients(model: nn.Module, *, quantizer: Quantizer, method: str, world_size: int) -> int:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    flat = torch.cat([parameter.grad.detach().reshape(-1) for parameter in parameters])
    if world_size > 1:
        qall_reduce(flat, op="mean", quantizer=quantizer, method=method, keep_self=False)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
        offset += count
    return flat.numel()


def train(args: argparse.Namespace) -> None:
    rank, world_size, device = setup(args.seed)
    model_dtype = torch.float16 if args.model_dtype == "fp16" else torch.float32
    model = SyntheticMLP(
        input_dim=args.input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=args.output_dim,
    ).to(device=device, dtype=model_dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    quantizer = Quantizer(args.group_size, -1, args.bit, 0, False, args.model_dtype)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    measured_step_times: list[float] = []
    sync_numel = 0
    start = time.perf_counter()
    for step in range(args.steps):
        inputs = torch.randn(args.batch_size_per_rank, args.input_dim, device=device, dtype=model_dtype)
        targets = torch.randn(args.batch_size_per_rank, args.output_dim, device=device, dtype=model_dtype)
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at rank={rank} step={step}")
        loss.backward()
        sync_numel = synchronize_gradients(model, quantizer=quantizer, method=args.legacy_method, world_size=world_size)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - step_start) * 1000
        losses.append(float(loss.detach()))
        if step >= args.warmup_steps:
            measured_step_times.append(step_ms)

    loss_total = torch.tensor([sum(losses), len(losses)], device=device, dtype=torch.float64)
    measured_total = torch.tensor([sum(measured_step_times), len(measured_step_times)], device=device, dtype=torch.float64)
    memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**20], device=device, dtype=torch.float64)
    dist.all_reduce(loss_total)
    dist.all_reduce(measured_total)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)

    if rank == 0:
        avg_step_ms = float(measured_total[0] / measured_total[1])
        result = {
            "mode": "legacy_ccdl",
            "world_size": world_size,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "global_batch_size": args.batch_size_per_rank * world_size,
            "input_dim": args.input_dim,
            "width": args.width,
            "depth": args.depth,
            "output_dim": args.output_dim,
            "parameter_count": parameter_count,
            "model_dtype": args.model_dtype,
            "legacy_method": args.legacy_method,
            "bit": args.bit,
            "group_size": args.group_size,
            "sync_numel": sync_numel,
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
