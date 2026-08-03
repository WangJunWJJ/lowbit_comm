from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import (
    CompressionConfig,
    compile_dynamic_all_gather,
    compressed_all_gather_dynamic,
)
from ccdl_comm.cuda.dynamic_gather_executor import DynamicGatherExecutorCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-numel", type=int, default=524_288)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--metadata-protocol",
        choices=("object_v1", "tensor_v1", "auto"),
        default="object_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)
    numel = args.base_numel + rank * args.group_size
    torch.manual_seed(20260729 + rank)
    local = torch.randn(numel, device=device, dtype=dtype)
    cache = DynamicGatherExecutorCache(max_entries=2)
    executor = compile_dynamic_all_gather(
        shape_class=(args.base_numel + (world_size - 1) * args.group_size,),
        config=config,
        dtype=args.dtype,
        cache=cache,
        metadata_protocol=args.metadata_protocol,
    )
    gathered = compressed_all_gather_dynamic(
        local,
        config=config,
        dtype=args.dtype,
        compiled_executor=executor,
    )

    references = []
    for index in range(world_size):
        torch.manual_seed(20260729 + index)
        references.append(torch.randn(args.base_numel + index * args.group_size, device=device, dtype=dtype))
    errors = [relative_l2(ref, got) for ref, got in zip(references, gathered, strict=True)]
    max_error = torch.stack(errors).max()
    shapes = [tuple(tensor.shape) for tensor in gathered]
    boundary_values = (0, 63, 64, 65)
    boundary_shapes = tuple(
        tuple(
            boundary_values[(round_index * world_size + index) % len(boundary_values)]
            for index in range(world_size)
        )
        for round_index in range(math.ceil(len(boundary_values) / world_size))
    )
    boundary_executor = compile_dynamic_all_gather(
        shape_class=(65,),
        config=config,
        dtype=args.dtype,
        cache=cache,
        metadata_protocol=args.metadata_protocol,
    )
    boundary_results = []
    boundary_errors = []
    for round_shapes in boundary_shapes:
        local_numel = round_shapes[rank]
        boundary_local = (
            torch.arange(local_numel, device=device, dtype=dtype) + rank * 0.25
        )
        boundary_gathered = compressed_all_gather_dynamic(
            boundary_local,
            config=config,
            dtype=args.dtype,
            compiled_executor=boundary_executor,
        )
        expected = [
            torch.arange(size, device=device, dtype=dtype) + index * 0.25
            for index, size in enumerate(round_shapes)
        ]
        boundary_results.append([tuple(tensor.shape) for tensor in boundary_gathered])
        boundary_errors.extend(
            relative_l2(reference, candidate)
            for reference, candidate in zip(expected, boundary_gathered, strict=True)
        )
    boundary_max_error = torch.stack(boundary_errors).max()
    summary = {
        "world_size": world_size,
        "base_numel": args.base_numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "shapes": shapes,
        "max_relative_l2": float(max_error),
        "boundary_shapes": boundary_results,
        "boundary_max_relative_l2": float(boundary_max_error),
        "compiled": True,
        "metadata_protocol_version": executor.metadata_protocol_version,
        "metadata_protocol_requested": args.metadata_protocol,
        "metadata_protocol_executed": executor.metadata_protocol,
        "metadata_protocol_fallback_reason": executor.execution_info.details[
            "metadata_protocol_fallback_reason"
        ],
        "shape_class_cache_entries": len(cache),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return (reference.float() - candidate.float()).norm() / reference.float().norm().clamp_min(1e-12)


if __name__ == "__main__":
    main()
