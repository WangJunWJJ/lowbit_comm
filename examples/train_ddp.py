"""End-to-end DDP training comparison with warmup-time bucket compilation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lowbit_comm.adapters.ddp import GradientFeedbackState, create_ddp_hook
from lowbit_comm.backends.cuda import CudaBackend, load_cuda_extension
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
class TrainingResult:
    mode: str
    samples_per_second: float
    step_p50_ms: float
    step_p95_ms: float
    loss_start: float
    loss_end: float
    rank_weight_gap: float


class ResidualBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)
        self.scale = 0.1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * torch.nn.functional.gelu(self.linear(value))


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _algorithm(mode: str) -> object:
    return {
        "compressed_all_gather": CompressedAllGather(),
        "compressed_rs_ag": CompressedReduceScatterAllGather(),
    }[mode]


def _register_compiled_hook(
    ddp: DistributedDataParallel,
    *,
    mode: str,
    rank: int,
    world_size: int,
    backend: CudaBackend,
) -> None:
    hooks: dict[tuple[int, int], object] = {}
    feedback = GradientFeedbackState()

    def hook(_unused_state: object, bucket: dist.GradBucket) -> torch.futures.Future:
        buffer = bucket.buffer()
        key = (bucket.index(), int(buffer.numel()))
        compiled_hook = hooks.get(key)
        if compiled_hook is None:
            program = CommunicationProgram(
                operation=ReduceMean(),
                output=FullTensor(DataType.FP16),
                wire=QuantizedWire(8, 64, compact=False),
                algorithm=_algorithm(mode),
            )
            context = CompileContext(
                rank=rank,
                world_size=world_size,
                shape=tuple(buffer.shape),
                dtype=DataType.FP16,
                device_type="cuda",
                device_architecture="sm86",
                topology_signature="single_node_pcie",
            )
            executable = backend.compile(
                backend.lower(program, context, RuntimeBindings(process_group=None))
            )
            compiled_hook = create_ddp_hook(
                executable,
                state=feedback,
                future_factory=torch.futures.Future,
                bucket_type=dist.GradBucket,
                return_type=torch.futures.Future[torch.Tensor],
            )
            hooks[key] = compiled_hook
        return compiled_hook(None, bucket)  # type: ignore[operator]

    hook.__annotations__ = {
        "_unused_state": object,
        "bucket": dist.GradBucket,
        "return": torch.futures.Future[torch.Tensor],
    }
    ddp.register_comm_hook(None, hook)


def _model(width: int, depth: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        *(ResidualBlock(width) for _ in range(depth))
    ).cuda().half()


def _train_mode(
    mode: str,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    width: int,
    depth: int,
    batch_size: int,
    warmup: int,
    steps: int,
    bucket_cap_mb: int,
    backend: CudaBackend,
) -> TrainingResult:
    torch.manual_seed(17)
    model = _model(width, depth)
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=True,
    )
    if mode != "native":
        _register_compiled_hook(
            ddp,
            mode=mode,
            rank=rank,
            world_size=world_size,
            backend=backend,
        )
    generator = torch.Generator(device="cuda").manual_seed(1000 + rank)
    inputs = torch.randn(
        batch_size,
        width,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    targets = inputs.mul(0.5)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
    losses: list[float] = []
    samples: list[float] = []
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = ddp(inputs)
        loss = torch.nn.functional.mse_loss(output.float(), targets.float())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if step >= warmup:
            losses.append(float(loss.item()))
            samples.append(elapsed_ms)
    timings = torch.tensor(samples, device="cuda", dtype=torch.float64)
    dist.all_reduce(timings, op=dist.ReduceOp.MAX)
    samples = timings.cpu().tolist()
    checksum = sum(parameter.float().sum() for parameter in model.parameters())
    checksums = [torch.empty_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)
    rank_weight_gap = max(float((checksums[0] - item).abs()) for item in checksums)
    p50 = _percentile(samples, 50)
    del ddp, model, optimizer
    torch.cuda.empty_cache()
    return TrainingResult(
        mode=mode,
        samples_per_second=world_size * batch_size * 1000.0 / p50,
        step_p50_ms=p50,
        step_p95_ms=_percentile(samples, 95),
        loss_start=losses[0],
        loss_end=losses[-1],
        rank_weight_gap=rank_weight_gap,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["native", "compressed_all_gather", "compressed_rs_ag"],
    )
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--bucket-cap-mb", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    backend = CudaBackend(extension_status=status)
    results: dict[str, list[TrainingResult]] = {mode: [] for mode in args.modes}
    for round_index in range(args.rounds):
        order = args.modes if round_index % 2 == 0 else list(reversed(args.modes))
        for mode in order:
            dist.barrier()
            results[mode].append(
                _train_mode(
                    mode,
                    rank=rank,
                    local_rank=local_rank,
                    world_size=world_size,
                    width=args.width,
                    depth=args.depth,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    steps=args.steps,
                    bucket_cap_mb=args.bucket_cap_mb,
                    backend=backend,
                )
            )
    if rank == 0:
        summary = {
            mode: {
                "samples_per_second": statistics.median(
                    item.samples_per_second for item in values
                ),
                "step_p50_ms": statistics.median(item.step_p50_ms for item in values),
                "step_p95_ms": statistics.median(item.step_p95_ms for item in values),
                "loss_start": statistics.median(item.loss_start for item in values),
                "loss_end": statistics.median(item.loss_end for item in values),
                "rank_weight_gap": max(item.rank_weight_gap for item in values),
            }
            for mode, values in results.items()
        }
        print(json.dumps({"world_size": world_size, "results": summary}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
