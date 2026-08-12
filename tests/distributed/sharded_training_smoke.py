"""Two-rank correctness oracle for exact ReducedShard SGD consumption."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from examples.training.sharded_sgd import (
    TorchShardedSgdConsumer,
    compile_torch_shard_layout,
    exact_mean_reduce_scatter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--steps", type=int, default=3)
    return parser


def _flat_parameters(parameters: tuple[Any, ...]) -> Any:
    return torch.cat(tuple(parameter.detach().reshape(-1) for parameter in parameters))


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1:
        raise ValueError("steps must be >= 1")
    dist.init_process_group(args.backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("sharded training smoke requires exactly two ranks")
        if args.backend == "nccl":
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        parameters = (
            torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            ),
            torch.nn.Parameter(torch.tensor([5.0], device=device)),
        )
        layout = compile_torch_shard_layout(
            parameters,
            rank=rank,
            world_size=world_size,
        )
        consumer = TorchShardedSgdConsumer(
            parameters,
            layout=layout,
            learning_rate=0.1,
            all_gather_into_tensor=dist.all_gather_into_tensor,
            torch=torch,
        )
        initial_pointers = consumer.buffer_pointers()
        local_gradient = (
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
            if rank == 0
            else torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=device)
        )

        for _ in range(args.steps):
            cursor = 0
            for parameter in parameters:
                count = parameter.numel()
                parameter.grad = local_gradient[cursor : cursor + count].reshape_as(
                    parameter
                )
                cursor += count
            flat_gradients = consumer.flatten_gradients()
            reduced = exact_mean_reduce_scatter(
                flat_gradients,
                out=consumer.reduced_output(),
                layout=layout,
                reduce_scatter_tensor=lambda output, source: dist.reduce_scatter_tensor(
                    output,
                    source,
                    op=dist.ReduceOp.SUM,
                ),
            )
            assert reduced.logical_range == layout.logical_range
            assert reduced.valid_numel == layout.valid_numel
            consumer.consume(reduced)
            assert consumer.buffer_pointers() == initial_pointers

        expected = torch.arange(1.0, 6.0, device=device) - 0.3 * args.steps
        actual = _flat_parameters(parameters)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
        gathered = [torch.empty_like(actual) for _ in range(world_size)]
        dist.all_gather(gathered, actual)
        max_rank_difference = max(
            float((candidate - gathered[0]).abs().max()) for candidate in gathered
        )
        if max_rank_difference != 0.0:
            raise AssertionError(
                f"rank parameters differ by {max_rank_difference:.9g}"
            )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "world_size": world_size,
                        "steps": args.steps,
                        "max_rank_difference": max_rank_difference,
                        "buffer_reuse_stable": True,
                        "logical_ranges": [[0, 3], [3, 5]],
                        "passed": True,
                    },
                    sort_keys=True,
                )
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
