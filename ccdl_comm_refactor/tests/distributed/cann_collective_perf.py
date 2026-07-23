from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from ccdl_comm import CompressionConfig, compressed_all_reduce
from ccdl_comm.ascend.codec import dequantize_tensor_cann, quantize_tensor_cann
from ccdl_comm.ascend.loader import load_cann_extension
from ccdl_comm.collectives.all_reduce import _make_payload_all_gather
from ccdl_comm.communication.torch_transport import make_torch_all_gather


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=1_048_576)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2031)
    return parser.parse_args()


def setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    return dist.get_rank(), dist.get_world_size(), torch.device("npu", local_rank)


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def benchmark(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.float() - candidate.float()).norm() / reference.float().norm())


def run() -> None:
    args = parse_args()
    rank, world_size, device = setup()
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
    extension_status = load_cann_extension()
    if not extension_status.available:
        raise RuntimeError(extension_status.reason or "ccdl_cann_ops is not available")
    payload_all_gather = _make_payload_all_gather(make_torch_all_gather())

    def ccdl_cann_once() -> None:
        compressed_all_reduce(
            source.clone(),
            config=config,
            op="mean",
            strategy="all_gather",
            dtype=args.dtype,
            quantize=lambda tensor, active_config: quantize_tensor_cann(
                tensor, active_config, extension_status=extension_status
            ),
            dequantize=lambda payload, shape, active_config, active_dtype: dequantize_tensor_cann(
                payload, shape, active_config, active_dtype, extension_status=extension_status
            ),
            all_gather=payload_all_gather,
        )

    torch_ms = benchmark(torch_all_reduce_once, warmup=args.warmup, repeat=args.repeat)
    ccdl_ms = benchmark(ccdl_cann_once, warmup=args.warmup, repeat=args.repeat)
    ccdl_result = compressed_all_reduce(
        source.clone(),
        config=config,
        op="mean",
        strategy="all_gather",
        dtype=args.dtype,
        quantize=lambda tensor, active_config: quantize_tensor_cann(
            tensor, active_config, extension_status=extension_status
        ),
        dequantize=lambda payload, shape, active_config, active_dtype: dequantize_tensor_cann(
            payload, shape, active_config, active_dtype, extension_status=extension_status
        ),
        all_gather=payload_all_gather,
    )
    torch.npu.synchronize()
    summary = {
        "backend": "hccl",
        "device_type": "npu",
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "world_size": world_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "torch_all_reduce_ms": torch_ms,
        "ccdl_cann_ms": ccdl_ms,
        "latency_ratio_ccdl_over_torch": ccdl_ms / torch_ms,
        "relative_l2": relative_l2(baseline_reference, ccdl_result),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
