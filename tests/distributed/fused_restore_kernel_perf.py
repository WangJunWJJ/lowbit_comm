"""Isolate CUDA operator costs in compressed ReducedShard restoration."""

from __future__ import annotations

import argparse
import json

import torch

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    dequantize_reduce_tensors,
    dequantize_tensor,
    inplace_dequantize_gathered,
    inplace_dequantize_reduce_mean_requantize,
    quantize_tensor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-numel", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def _cuda_event_ms(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def main() -> None:
    args = _parse_args()
    if args.shard_numel <= 0 or args.shard_numel % 64:
        raise ValueError("shard-numel must be a positive multiple of 64")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    status = load_cuda_extension()
    if not torch.cuda.is_available() or not status.available:
        raise RuntimeError(status.reason or "CCDL CUDA extension is unavailable")
    config = CompressionConfig(bit=8, group_size=64, topk=0, compact=False)
    sources = [
        torch.randn(args.shard_numel, device="cuda", dtype=torch.float16)
        for _ in range(args.world_size)
    ]
    payloads = [quantize_tensor(source, config, extension_status=status) for source in sources]
    reduced = torch.empty_like(sources[0])
    legacy_requantized = allocate_quantized_buffer(reduced, config, dtype="fp16")
    payload_numel = int(legacy_requantized.numel())
    payload_stride = ((payload_numel + 15) // 16) * 16
    fused_requantized = torch.empty(payload_stride, device="cuda", dtype=torch.uint8)

    def legacy_requantize_chain() -> None:
        dequantize_reduce_tensors(
            payloads,
            (args.shard_numel,),
            config,
            dtype="fp16",
            extension_status=status,
            output=reduced,
            reduce="mean",
        )
        quantize_tensor(
            reduced,
            config,
            extension_status=status,
            output=legacy_requantized,
        )

    def fused_requantize() -> None:
        if not inplace_dequantize_reduce_mean_requantize(
            payloads,
            fused_requantized,
            config,
            dtype="fp16",
            extension_status=status,
            divisor=args.world_size,
        ):
            raise RuntimeError("fused requantize fast path was rejected")

    legacy_requantize_chain()
    fused_requantize()
    torch.cuda.synchronize()
    if not torch.equal(legacy_requantized, fused_requantized[:payload_numel]):
        raise AssertionError("fused requantize output differs from the legacy operator chain")

    gathered = torch.zeros(args.world_size * payload_stride, device="cuda", dtype=torch.uint8)
    for rank in range(args.world_size):
        start = rank * payload_stride
        gathered[start : start + payload_numel].copy_(fused_requantized[:payload_numel])
    legacy_restored = torch.empty(args.world_size * args.shard_numel, device="cuda", dtype=torch.float16)
    fused_restored = torch.empty_like(legacy_restored)

    def legacy_gathered_dequantize() -> None:
        for rank in range(args.world_size):
            payload = gathered.narrow(0, rank * payload_stride, payload_numel)
            output = legacy_restored.narrow(0, rank * args.shard_numel, args.shard_numel)
            dequantize_tensor(
                payload,
                (args.shard_numel,),
                config,
                dtype="fp16",
                extension_status=status,
                output=output,
            )

    def fused_gathered_dequantize() -> None:
        if not inplace_dequantize_gathered(
            gathered,
            fused_restored,
            config,
            dtype="fp16",
            extension_status=status,
            world_size=args.world_size,
            payload_numel=payload_numel,
            payload_stride=payload_stride,
            shard_numel=args.shard_numel,
        ):
            raise RuntimeError("fused gathered-dequantize fast path was rejected")

    legacy_gathered_dequantize()
    fused_gathered_dequantize()
    torch.cuda.synchronize()
    if not torch.equal(legacy_restored, fused_restored):
        raise AssertionError("fused gathered dequantization differs from the rank loop")

    legacy_requantize_ms = _cuda_event_ms(
        legacy_requantize_chain,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused_requantize_ms = _cuda_event_ms(
        fused_requantize,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    legacy_gathered_ms = _cuda_event_ms(
        legacy_gathered_dequantize,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused_gathered_ms = _cuda_event_ms(
        fused_gathered_dequantize,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    evidence = {
        "world_size": args.world_size,
        "shard_numel": args.shard_numel,
        "legacy_requantize_chain_ms": legacy_requantize_ms,
        "fused_requantize_ms": fused_requantize_ms,
        "requantize_speedup": legacy_requantize_ms / fused_requantize_ms,
        "legacy_gathered_dequantize_ms": legacy_gathered_ms,
        "fused_gathered_dequantize_ms": fused_gathered_ms,
        "gathered_dequantize_speedup": legacy_gathered_ms / fused_gathered_ms,
    }
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
