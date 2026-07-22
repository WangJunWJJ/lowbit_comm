from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig


class TinyCifarNet(nn.Module):
    """Small CNN used for end-to-end CIFAR10 DDP communication validation."""

    def __init__(self, num_classes: int = 10) -> None:
        """Create a compact image classifier.

        Args:
            num_classes: Number of output classes.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of CIFAR-like images.

        Args:
            images: Tensor shaped ``[batch, 3, 32, 32]``.

        Returns:
            Class logits shaped ``[batch, num_classes]``.
        """

        features = self.features(images).flatten(1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    """Parse command line options for the distributed CIFAR10 comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "ccdl"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size-per-rank", type=int, default=128)
    parser.add_argument("--workers-per-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--strategy", choices=("all_gather", "all_reduce"), default="all_gather")
    return parser.parse_args()


def setup_distributed(seed: int) -> tuple[int, int, torch.device]:
    """Initialize NCCL distributed state and deterministic seeds.

    Args:
        seed: Base random seed.

    Returns:
        A tuple of rank, world size, and current CUDA device.
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rank, dist.get_world_size(), torch.device("cuda", local_rank)


def build_loaders(args: argparse.Namespace, rank: int, world_size: int) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    """Build distributed CIFAR10 train and validation loaders.

    Args:
        args: Parsed command line options.
        rank: Current distributed rank.
        world_size: Number of distributed ranks.

    Returns:
        Train loader, validation loader, and train sampler.
    """

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if rank == 0:
        datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_transform)
        datasets.CIFAR10(args.data_root, train=False, download=True, transform=val_transform)
    dist.barrier()
    train_set = datasets.CIFAR10(args.data_root, train=True, download=False, transform=train_transform)
    val_set = datasets.CIFAR10(args.data_root, train=False, download=False, transform=val_transform)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    loader_args = {
        "batch_size": args.batch_size_per_rank,
        "num_workers": args.workers_per_rank,
        "pin_memory": True,
        "persistent_workers": args.workers_per_rank > 0,
    }
    return (
        DataLoader(train_set, sampler=train_sampler, **loader_args),
        DataLoader(val_set, sampler=val_sampler, **loader_args),
        train_sampler,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> DistributedDataParallel:
    """Build the DDP model and optionally attach the CCDL communication hook."""

    local_rank = int(os.environ["LOCAL_RANK"])
    model = TinyCifarNet().to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    if args.mode == "ccdl":
        config = CompressionConfig(bit=args.bit, group_size=args.group_size, error_feedback=True)
        hook = create_ddp_comm_hook(config, dtype="auto", strategy=args.strategy, reduce="mean")
        ddp_model.register_comm_hook(state=None, hook=hook)
    return ddp_model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate distributed validation loss and top-1 accuracy."""

    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        totals[0] += nn.functional.cross_entropy(logits, targets, reduction="sum")
        totals[1] += (logits.argmax(1) == targets).sum()
        totals[2] += targets.numel()
    dist.all_reduce(totals)
    return float(totals[0] / totals[2]), float(100 * totals[1] / totals[2])


def train(args: argparse.Namespace) -> None:
    """Run CIFAR10 training and write rank-zero metrics as JSON."""

    rank, world_size, device = setup_distributed(args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)
    model = build_model(args, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    measured_step_times: list[float] = []
    train_start = time.perf_counter()
    step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at rank={rank} step={step}")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            step_ms = (time.perf_counter() - step_start) * 1000
            losses.append(float(loss.detach()))
            if step >= args.warmup_steps:
                measured_step_times.append(step_ms)
            step += 1
    train_loss = torch.tensor([sum(losses), len(losses)], device=device, dtype=torch.float64)
    dist.all_reduce(train_loss)
    val_loss, val_top1 = evaluate(model, val_loader, device)
    elapsed_s = time.perf_counter() - train_start
    local_step_ms = torch.tensor([sum(measured_step_times), len(measured_step_times)], device=device, dtype=torch.float64)
    local_memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**20], device=device, dtype=torch.float64)
    dist.all_reduce(local_step_ms)
    dist.all_reduce(local_memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "mode": args.mode,
            "strategy": args.strategy if args.mode == "ccdl" else "ddp_default",
            "bit": args.bit if args.mode == "ccdl" else None,
            "group_size": args.group_size if args.mode == "ccdl" else None,
            "error_feedback": args.mode == "ccdl",
            "epochs": args.epochs,
            "steps_per_rank": step,
            "global_batch_size": args.batch_size_per_rank * world_size,
            "world_size": world_size,
            "seed": args.seed,
            "train_loss": float(train_loss[0] / train_loss[1]),
            "val_loss": val_loss,
            "val_top1": val_top1,
            "avg_step_ms": float(local_step_ms[0] / local_step_ms[1]),
            "images_per_s": float(args.batch_size_per_rank * world_size / (local_step_ms[0] / local_step_ms[1] / 1000)),
            "peak_memory_mb_max_rank": float(local_memory[0]),
            "elapsed_s_rank0": elapsed_s,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
