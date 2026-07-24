from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_dequantized_buffer,
    allocate_quantized_buffer,
    dequantize_tensor,
    quantize_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=4_194_304)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "fp32": torch.float32}[name]


def _benchmark(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason or "ccdl_cuda_ops is not available")

    device = torch.device("cuda", 0)
    tensor = torch.randn(args.numel, device=device, dtype=_torch_dtype(args.dtype))
    config = CompressionConfig(bit=args.bit, group_size=args.group_size, compact=args.compact)
    quantized_reference = quantize_tensor(tensor, config, extension_status=status)
    quantized_output = allocate_quantized_buffer(tensor, config, dtype=args.dtype)
    dequantized_output = allocate_dequantized_buffer(tensor, tensor.shape, config)

    alloc_quant_ms = _benchmark(
        lambda: quantize_tensor(tensor, config, extension_status=status),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    inplace_quant_ms = _benchmark(
        lambda: quantize_tensor(tensor, config, extension_status=status, output=quantized_output),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    alloc_dequant_ms = _benchmark(
        lambda: dequantize_tensor(quantized_reference, tensor.shape, config, dtype=args.dtype, extension_status=status),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    inplace_dequant_ms = _benchmark(
        lambda: dequantize_tensor(
            quantized_reference,
            tensor.shape,
            config,
            dtype=args.dtype,
            extension_status=status,
            output=dequantized_output,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    restored = dequantize_tensor(
        quantized_reference,
        tensor.shape,
        config,
        dtype=args.dtype,
        extension_status=status,
        output=dequantized_output,
    )
    torch.cuda.synchronize()
    relative_l2 = float((tensor.float() - restored.float()).norm() / tensor.float().norm())
    result = {
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "compact": args.compact,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "alloc_quant_ms": alloc_quant_ms,
        "inplace_quant_ms": inplace_quant_ms,
        "quant_speedup": alloc_quant_ms / inplace_quant_ms,
        "alloc_dequant_ms": alloc_dequant_ms,
        "inplace_dequant_ms": inplace_dequant_ms,
        "dequant_speedup": alloc_dequant_ms / inplace_dequant_ms,
        "relative_l2": relative_l2,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
