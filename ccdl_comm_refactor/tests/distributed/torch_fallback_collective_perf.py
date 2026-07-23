from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compressed_all_reduce
from ccdl_comm.quantization.torch_fallback import dequantize_tensor_fallback, quantize_tensor_fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--backend", choices=("hccl", "nccl", "gloo"), default="hccl")
    parser.add_argument("--device-type", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def setup(backend: str, device_type: str) -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    elif device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend)
    return dist.get_rank(), dist.get_world_size(), device


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def synchronize(device_type: str) -> None:
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


def benchmark(fn, *, warmup: int, repeat: int, device_type: str) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device_type)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize(device_type)
    return (time.perf_counter() - start) * 1000 / repeat


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.float() - candidate.float()).norm() / reference.float().norm())


def run() -> None:
    args = parse_args()
    rank, world_size, device = setup(args.backend, args.device_type)
    torch.manual_seed(args.seed + rank)
    dtype = dtype_from_name(args.dtype)
    source = torch.randn(args.numel, device=device, dtype=dtype)
    baseline_reference = source.clone()
    dist.all_reduce(baseline_reference, op=dist.ReduceOp.SUM)
    baseline_reference /= world_size

    def torch_all_reduce_once() -> None:
        tensor = source.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size

    config = CompressionConfig(bit=args.bit, group_size=args.group_size, quant_type="linear")

    def ccdl_torch_fallback_once() -> None:
        compressed_all_reduce(
            source.clone(),
            config=config,
            op="mean",
            strategy="all_gather",
            dtype=args.dtype,
            quantize=quantize_tensor_fallback,
            dequantize=dequantize_tensor_fallback,
        )

    torch_ms = benchmark(torch_all_reduce_once, warmup=args.warmup, repeat=args.repeat, device_type=args.device_type)
    ccdl_ms = benchmark(ccdl_torch_fallback_once, warmup=args.warmup, repeat=args.repeat, device_type=args.device_type)
    ccdl_result = compressed_all_reduce(
        source.clone(),
        config=config,
        op="mean",
        strategy="all_gather",
        dtype=args.dtype,
        quantize=quantize_tensor_fallback,
        dequantize=dequantize_tensor_fallback,
    )
    synchronize(args.device_type)
    summary = {
        "backend": args.backend,
        "device_type": args.device_type,
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "world_size": world_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "torch_all_reduce_ms": torch_ms,
        "ccdl_torch_fallback_ms": ccdl_ms,
        "latency_ratio_ccdl_over_torch": ccdl_ms / torch_ms,
        "relative_l2": relative_l2(baseline_reference, ccdl_result),
        "torch": torch.__version__,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
