import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import MAIN_VARIANTS, RunConfig
from .data import build_loaders
from .logging_utils import JsonlLogger, cuda_elapsed_ms
from .model import build_model
from .sync import FlatGradientSynchronizer


def _setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    import ccdl.comm as comm

    comm.init()
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _environment():
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "world_size": dist.get_world_size(),
    }


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        totals[0] += torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        totals[1] += (logits.argmax(1) == targets).sum()
        totals[2] += targets.numel()
    dist.all_reduce(totals)
    return float(totals[0] / totals[2]), float(100 * totals[1] / totals[2])


def run(config: RunConfig, data_root: Path, resume: bool):
    rank, world_size, device = _setup()
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(config.output_dir / "metrics.jsonl", rank)
    model = build_model().to(device)
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    initial_hash = hashlib.sha256(
        b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    train_loader, val_loader, sampler = build_loaders(data_root, config, rank, world_size)
    synchronizer = FlatGradientSynchronizer(
        model, config.variant, config.bit, config.topk, config.group_size
    )
    start_epoch = 0
    steps = 0
    checkpoint = config.output_dir / "last.pt"
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, steps = state["epoch"] + 1, state["steps"]
    if rank == 0:
        (config.output_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )
        (config.output_dir / "environment.json").write_text(
            json.dumps({**_environment(), "initial_parameter_sha256": initial_hash}, indent=2),
            encoding="utf-8",
        )
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        totals = {key: 0.0 for key in ("loss", "correct", "samples", "forward_ms", "backward_ms", "sync_ms", "optimizer_ms", "step_ms")}
        for images, targets in train_loader:
            step_start = time.perf_counter()
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def forward():
                logits = model(images)
                return logits, torch.nn.functional.cross_entropy(logits, targets)

            (logits, loss), forward_ms = cuda_elapsed_ms(forward)
            _, backward_ms = cuda_elapsed_ms(loss.backward)
            flat = synchronizer.pack()
            grad_norm = float(flat.norm())
            if not torch.isfinite(flat).all():
                raise FloatingPointError(f"non-finite gradient at epoch={epoch} step={steps}")
            _, sync_ms = cuda_elapsed_ms(lambda: synchronizer.synchronize(flat))
            synchronizer.unpack(flat)
            _, optimizer_ms = cuda_elapsed_ms(optimizer.step)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1000
            batch = targets.numel()
            totals["loss"] += float(loss) * batch
            totals["correct"] += int((logits.argmax(1) == targets).sum())
            totals["samples"] += batch
            for key, value in (("forward_ms", forward_ms), ("backward_ms", backward_ms), ("sync_ms", sync_ms), ("optimizer_ms", optimizer_ms), ("step_ms", step_ms)):
                if steps >= 20:
                    totals[key] += value
            steps += 1
            logger.emit("step", epoch=epoch, optimizer_steps=steps, loss=float(loss), grad_norm=grad_norm, forward_ms=forward_ms, backward_ms=backward_ms, sync_ms=sync_ms, optimizer_ms=optimizer_ms, step_ms=step_ms)
        scheduler.step()
        reduced = torch.tensor([totals["loss"], totals["correct"], totals["samples"]], device=device, dtype=torch.float64)
        dist.all_reduce(reduced)
        val_loss, val_top1 = _evaluate(model, val_loader, device)
        measured_steps = max(1, len(train_loader) - (20 if epoch == 0 else 0))
        epoch_row = {
            "epoch": epoch,
            "optimizer_steps": steps,
            "wall_s": time.perf_counter() - wall_start,
            "train_loss": float(reduced[0] / reduced[2]),
            "train_top1": float(100 * reduced[1] / reduced[2]),
            "val_loss": val_loss,
            "val_top1": val_top1,
            "images_per_s": config.batch_size_per_rank * world_size / (totals["step_ms"] / measured_steps / 1000),
            "sync_ms": totals["sync_ms"] / measured_steps,
            "step_ms": totals["step_ms"] / measured_steps,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.emit("epoch", **epoch_row)
        if rank == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "steps": steps}, checkpoint)
            csv_path = config.output_dir / "epochs.csv"
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=epoch_row.keys())
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(epoch_row)
    if rank == 0:
        (config.output_dir / "complete.json").write_text(json.dumps({"epochs": config.epochs, "steps": steps}), encoding="utf-8")
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=MAIN_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size-per-rank", type=int, default=128)
    parser.add_argument("--workers-per-rank", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    bit, topk = MAIN_VARIANTS[args.variant]
    config = RunConfig(args.variant, args.seed, args.output_dir, epochs=args.epochs, batch_size_per_rank=args.batch_size_per_rank, workers_per_rank=args.workers_per_rank, bit=bit, topk=topk)
    run(config, args.data_root, args.resume)


if __name__ == "__main__":
    main()
