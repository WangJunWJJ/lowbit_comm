import math
import argparse
import json
import os
import random
import time
from pathlib import Path


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def token_accuracy_counts(logits, labels) -> tuple[int, int]:
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            predictions = logits.argmax(dim=-1)
            valid = labels != -100
            return int(((predictions == labels) & valid).sum().item()), int(valid.sum().item())
    except ImportError:
        pass
    correct = total = 0
    for scores, label in zip(logits, labels):
        if label == -100:
            continue
        prediction = max(range(len(scores)), key=scores.__getitem__)
        correct += int(prediction == label)
        total += 1
    return correct, total


def _setup_distributed():
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    import ccdl.comm as comm
    comm.init()
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def _seed(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from .data import AlpacaDataset, ResponseOnlyCollator, load_alpaca, split_indices
    from .logging_utils import JsonlLogger, mark_completed
    from .model import load_qwen2_text_model
    from .sync import FlatGradientSynchronizer

    rank, world_size, device = _setup_distributed()
    _seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "metrics.jsonl") if rank == 0 else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_alpaca(args.data_path)
    train_indices, validation_indices = split_indices(len(records), 20260717, 0.05)
    train_data = AlpacaDataset(records, train_indices, tokenizer, args.max_length)
    validation_data = AlpacaDataset(records, validation_indices, tokenizer, args.max_length)
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    collator = ResponseOnlyCollator(tokenizer.pad_token_id)
    loader = DataLoader(train_data, batch_size=args.micro_batch_size, sampler=sampler, collate_fn=collator, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    model = load_qwen2_text_model(args.model_path, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, args.max_steps // 20), args.max_steps)
    synchronizer = FlatGradientSynchronizer(args.variant, world_size)
    if rank == 0:
        environment = {
            "arguments": vars(args),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world_size,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        (output_dir / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = 0
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def evaluate():
        model.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if batch_index >= args.eval_batches:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**batch)
                labels = batch["labels"][:, 1:]
                predictions = output.logits[:, :-1].argmax(-1)
                valid = labels != -100
                tokens = valid.sum()
                totals[0] += output.loss.double() * tokens
                totals[1] += ((predictions == labels) & valid).sum()
                totals[2] += tokens
        dist.all_reduce(totals)
        model.train()
        loss = float(totals[0] / totals[2])
        return loss, perplexity_from_loss(loss), float(totals[1] / totals[2])

    epoch = 0
    sampler.set_epoch(epoch)
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps:
                continue
            metrics = synchronizer.synchronize(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - wall_start
            if rank == 0:
                logger.emit("step", step=step, loss=float(output.loss), sync_ms=metrics.elapsed_ms,
                            communicated_bytes=metrics.communicated_bytes, wall_time_sec=elapsed,
                            tokens_per_sec=(args.micro_batch_size * world_size * args.max_length * args.gradient_accumulation_steps) / max(1e-9, elapsed / step),
                            peak_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20)
            if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
                val_loss, perplexity, accuracy = evaluate()
                if rank == 0:
                    logger.emit("eval", step=step, val_loss=val_loss, perplexity=perplexity,
                                token_accuracy=accuracy, wall_time_sec=time.perf_counter() - wall_start)
            if step >= args.max_steps:
                break
        epoch += 1
        sampler.set_epoch(epoch)
    if rank == 0:
        mark_completed(output_dir, {"variant": args.variant, "seed": args.seed, "steps": step,
                                    "wall_time_sec": time.perf_counter() - wall_start})
    dist.barrier()


def main():
    from .config import MAIN_VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=MAIN_VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
