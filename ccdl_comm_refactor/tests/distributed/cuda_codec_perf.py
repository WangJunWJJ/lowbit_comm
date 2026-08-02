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
    inplace_quantize_pack,
    quantize_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=4_194_304)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--compact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


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
    residual = torch.zeros_like(tensor) if args.residual else None
    prepared = torch.empty_like(tensor) if args.residual else None
    fused_metadata: dict[str, object] = {}
    if not inplace_quantize_pack(
        tensor,
        quantized_output,
        residual,
        config,
        fused_metadata,
        extension_status=status,
    ):
        raise RuntimeError("requested codec configuration does not support fused quant-pack")

    alloc_quant_ms = _benchmark(
        lambda: quantize_tensor(tensor, config, extension_status=status),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    legacy_inplace_quant_ms = _benchmark(
        lambda: status.module.inplace_quantize(
            tensor,
            quantized_output,
            config.group_size,
            config.topk,
            config.stochastic,
            config.bit,
            status.module.QuantType.Linear,
            config.compact,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    legacy_error_feedback_quant_ms = None
    if residual is not None:
        def legacy_error_feedback_quant() -> None:
            torch.add(tensor, residual, out=prepared)
            status.module.inplace_quantize(
                prepared,
                quantized_output,
                config.group_size,
                config.topk,
                config.stochastic,
                config.bit,
                status.module.QuantType.Linear,
                config.compact,
            )

        legacy_error_feedback_quant_ms = _benchmark(
            legacy_error_feedback_quant,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    allocated_before = torch.cuda.memory_allocated()
    fused_quant_pack_ms = _benchmark(
        lambda: inplace_quantize_pack(
            tensor,
            quantized_output,
            residual,
            config,
            fused_metadata,
            extension_status=status,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    allocated_after = torch.cuda.memory_allocated()
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
        quantized_output,
        tensor.shape,
        config,
        dtype=args.dtype,
        extension_status=status,
        output=dequantized_output,
    )
    torch.cuda.synchronize()
    reference_tensor = tensor if residual is None else tensor + residual
    tensor_fp32 = reference_tensor.float()
    restored_fp32 = restored.float()
    difference = tensor_fp32 - restored_fp32
    relative_l2 = float(difference.norm() / tensor_fp32.norm())
    max_abs_error = float(difference.abs().max()) if difference.numel() else 0.0
    rmse = float(difference.square().mean().sqrt()) if difference.numel() else 0.0
    non_finite = int((~torch.isfinite(restored_fp32)).sum())
    result = {
        "numel": args.numel,
        "dtype": args.dtype,
        "bit": args.bit,
        "group_size": args.group_size,
        "compact": args.compact,
        "residual": args.residual,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "alloc_quant_ms": alloc_quant_ms,
        "legacy_inplace_quant_ms": legacy_inplace_quant_ms,
        "legacy_error_feedback_quant_ms": legacy_error_feedback_quant_ms,
        "fused_quant_pack_ms": fused_quant_pack_ms,
        "inplace_quant_ms": fused_quant_pack_ms,
        "quant_speedup": (legacy_error_feedback_quant_ms or legacy_inplace_quant_ms) / fused_quant_pack_ms,
        "fused_quant_pack_used": fused_metadata.get("fused_quant_pack", False),
        "fused_quant_pack_allocated_bytes": allocated_after - allocated_before,
        "alloc_dequant_ms": alloc_dequant_ms,
        "inplace_dequant_ms": inplace_dequant_ms,
        "dequant_speedup": alloc_dequant_ms / inplace_dequant_ms,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "non_finite": non_finite,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
