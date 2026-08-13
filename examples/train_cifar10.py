"""Reproducible CIFAR-10/ResNet-18 time-to-quality benchmark.

Run this entry once per communication mode so each process group owns exactly
one training state.  Use the same seed and command-line options for fair runs.
The dataset is downloaded only by rank zero and then shared after a barrier.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms

from lowbit_comm.adapters.ddp import GradientFeedbackState, create_ddp_hook
from lowbit_comm.backends.cuda import CudaBackend, load_cuda_extension
from lowbit_comm.benchmarking import SCHEMA_VERSION, runtime_fingerprint
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    epoch_samples_per_second: float
    training_loss: float
    validation_accuracy: float
    validation_loss: float
    elapsed_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("native", "compressed_all_gather", "compressed_rs_ag"),
        default="native",
    )
    parser.add_argument("--dataset", choices=("cifar10", "fake"), default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    parser.add_argument("--fake-train-size", type=int, default=1024)
    parser.add_argument("--fake-validation-size", type=int, default=256)
    return parser.parse_args()


def _seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _algorithm(mode: str) -> object:
    return {
        "compressed_all_gather": CompressedAllGather(),
        "compressed_rs_ag": CompressedReduceScatterAllGather(),
    }[mode]


def _register_hook(
    ddp: DistributedDataParallel,
    *,
    mode: str,
    rank: int,
    world_size: int,
) -> None:
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    backend = CudaBackend(extension_status=status)
    feedback = GradientFeedbackState()
    hooks: dict[tuple[int, int, torch.dtype], Any] = {}

    def hook(_unused_state: object, bucket: dist.GradBucket) -> torch.futures.Future:
        buffer = bucket.buffer()
        key = (bucket.index(), int(buffer.numel()), buffer.dtype)
        compiled = hooks.get(key)
        if compiled is None:
            dtype = {
                torch.float16: DataType.FP16,
                torch.bfloat16: DataType.BF16,
                torch.float32: DataType.FP32,
            }.get(buffer.dtype)
            if dtype is None:
                raise TypeError(f"unsupported DDP bucket dtype: {buffer.dtype}")
            program = CommunicationProgram(
                operation=ReduceMean(),
                output=FullTensor(dtype),
                wire=QuantizedWire(8, 64, compact=False),
                algorithm=_algorithm(mode),
            )
            context = CompileContext(
                rank=rank,
                world_size=world_size,
                shape=tuple(buffer.shape),
                dtype=dtype,
                device_type="cuda",
                device_architecture="sm86",
                topology_signature="single_node_pcie",
            )
            executable = backend.compile(
                backend.lower(program, context, RuntimeBindings(process_group=None))
            )
            compiled = create_ddp_hook(
                executable,
                state=feedback,
                future_factory=torch.futures.Future,
                bucket_type=dist.GradBucket,
                return_type=torch.futures.Future[torch.Tensor],
            )
            hooks[key] = compiled
        return compiled(None, bucket)

    hook.__annotations__ = {
        "_unused_state": object,
        "bucket": dist.GradBucket,
        "return": torch.futures.Future[torch.Tensor],
    }
    ddp.register_comm_hook(None, hook)


def _loaders(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if args.dataset == "cifar10":
        if rank == 0:
            datasets.CIFAR10(args.data_root, train=True, download=True)
            datasets.CIFAR10(args.data_root, train=False, download=True)
        dist.barrier()
        train_set = datasets.CIFAR10(
            args.data_root, train=True, transform=train_transform, download=False
        )
        validation_set = datasets.CIFAR10(
            args.data_root, train=False, transform=validation_transform, download=False
        )
    else:
        train_set = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
            random_offset=args.seed,
        )
        validation_set = datasets.FakeData(
            size=args.fake_validation_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=validation_transform,
            random_offset=args.seed + args.fake_train_size,
        )
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    validation_sampler = DistributedSampler(
        validation_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set, sampler=train_sampler, drop_last=True, **loader_options
    )
    validation_loader = DataLoader(
        validation_set, sampler=validation_sampler, drop_last=False, **loader_options
    )
    return train_loader, validation_loader, train_sampler, validation_sampler


def _resnet18() -> torch.nn.Module:
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model.cuda()


def _train_epoch(
    ddp: DistributedDataParallel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    world_size: int,
) -> tuple[float, float]:
    ddp.train()
    aggregate = torch.zeros(2, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for inputs, labels in loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(ddp(inputs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        aggregate[0] += loss.detach().double() * labels.numel()
        aggregate[1] += labels.numel()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    dist.all_reduce(aggregate)
    elapsed_tensor = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    samples = float(aggregate[1].item())
    return float(aggregate[0].item() / samples), samples / float(elapsed_tensor.item())


@torch.no_grad()
def _validate(
    ddp: DistributedDataParallel,
    loader: DataLoader,
) -> tuple[float, float]:
    ddp.eval()
    aggregate = torch.zeros(3, device="cuda", dtype=torch.float64)
    for inputs, labels in loader:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = ddp(inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        aggregate[0] += loss.double() * labels.numel()
        aggregate[1] += (logits.argmax(dim=1) == labels).sum()
        aggregate[2] += labels.numel()
    dist.all_reduce(aggregate)
    samples = float(aggregate[2].item())
    return float(aggregate[0].item() / samples), float(aggregate[1].item() / samples)


def _rank_weight_gap(model: torch.nn.Module, world_size: int) -> float:
    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in model.parameters()]
    )
    gathered = [torch.empty_like(checksum) for _ in range(world_size)]
    dist.all_gather(gathered, checksum)
    reference = gathered[0]
    return max(float((reference - item).abs().max().item()) for item in gathered)


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    _seed_everything(args.seed, rank)
    train_loader, validation_loader, train_sampler, validation_sampler = _loaders(
        args, rank=rank, world_size=world_size
    )
    model = _resnet18()
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=args.bucket_cap_mb,
        gradient_as_bucket_view=True,
    )
    if args.mode != "native":
        _register_hook(ddp, mode=args.mode, rank=rank, world_size=world_size)
    optimizer = torch.optim.SGD(
        ddp.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[EpochResult] = []
    target_time: float | None = None
    run_started = time.perf_counter()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        training_loss, throughput = _train_epoch(
            ddp, train_loader, optimizer, scaler, world_size=world_size
        )
        validation_loss, validation_accuracy = _validate(ddp, validation_loader)
        scheduler.step()
        elapsed = time.perf_counter() - run_started
        if target_time is None and validation_accuracy >= args.target_accuracy:
            target_time = elapsed
        result = EpochResult(
            epoch=epoch + 1,
            epoch_samples_per_second=throughput,
            training_loss=training_loss,
            validation_accuracy=validation_accuracy,
            validation_loss=validation_loss,
            elapsed_seconds=elapsed,
        )
        history.append(result)
        if rank == 0:
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
    rank_weight_gap = _rank_weight_gap(model, world_size)
    if rank == 0:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": runtime_fingerprint(
                torch,
                local_rank=local_rank,
            ),
            "mode": args.mode,
            "dataset": args.dataset,
            "seed": args.seed,
            "global_batch_size": args.batch_size * world_size,
            "target_accuracy": args.target_accuracy,
            "time_to_target_seconds": target_time,
            "rank_weight_gap": rank_weight_gap,
            "epochs": [asdict(item) for item in history],
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        print(encoded, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
