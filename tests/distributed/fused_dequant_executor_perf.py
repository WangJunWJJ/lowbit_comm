from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig
from ccdl_comm.cuda.backend import CudaCommunicationBackend
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    dequantize_reduce_tensors,
    quantize_tensor,
    update_error_feedback_residual,
)
from tests.benchmarks.fused_dequant_executor_gate import measure_balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-mib", type=int, choices=(1, 16, 64), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _measure(operation, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        operation()
    end.record()
    end.synchronize()
    local_ms = torch.tensor([start.elapsed_time(end) / repeat], device="cuda", dtype=torch.float64)
    dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
    return float(local_ms.item())


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    difference = (candidate.float() - reference.float()).norm()
    denominator = reference.float().norm().clamp_min(1e-12)
    return float((difference / denominator).item())


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    numel = args.bucket_mib * 1024 * 1024 // torch.empty((), dtype=torch_dtype).element_size()
    config = CompressionConfig(bit=8, group_size=64, error_feedback=True)
    extension_status = load_cuda_extension()
    if not extension_status.available:
        raise RuntimeError(extension_status.reason or "CCDL CUDA extension is unavailable")

    source = torch.randn(numel, device=device, dtype=torch_dtype) * 0.1 + rank * 0.01
    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    reference.div_(world_size)

    send_baseline = allocate_quantized_buffer(source, config, dtype=args.dtype)
    recv_baseline = [torch.empty_like(send_baseline) for _ in range(world_size)]
    prepared_baseline = torch.empty_like(source)
    output_baseline = torch.empty_like(source)
    residual_baseline = torch.zeros_like(source)

    send_fused = torch.empty_like(send_baseline)
    recv_fused = [torch.empty_like(send_fused) for _ in range(world_size)]
    prepared_fused = torch.empty_like(source)
    output_fused = torch.empty_like(source)
    residual_fused = torch.zeros_like(source)

    executor = CudaCommunicationBackend(extension_status=extension_status).compile(
        CommunicationPlan("all_reduce", "all_gather", compression=config),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=str(device),
            shape=(numel,),
            dtype=args.dtype,
        ),
    )

    def baseline_once() -> None:
        torch.add(source, residual_baseline, out=prepared_baseline)
        quantize_tensor(
            prepared_baseline,
            config,
            extension_status=extension_status,
            output=send_baseline,
        )
        dist.all_gather(recv_baseline, send_baseline)
        dequantize_reduce_tensors(
            recv_baseline,
            (numel,),
            config,
            dtype=args.dtype,
            extension_status=extension_status,
            output=output_baseline,
            reduce="sum",
        )
        output_baseline.div_(world_size)
        update_error_feedback_residual(
            prepared_baseline,
            output_baseline,
            residual_baseline,
            extension_status=extension_status,
        )

    def fused_once() -> None:
        torch.add(source, residual_fused, out=prepared_fused)
        quantize_tensor(
            prepared_fused,
            config,
            extension_status=extension_status,
            output=send_fused,
        )
        dist.all_gather(recv_fused, send_fused)
        executor.run_precollected_payloads(
            recv_fused,
            prepared=prepared_fused,
            output=output_fused,
            residual=residual_fused,
        )

    residual_baseline.zero_()
    baseline_once()
    torch.cuda.synchronize()
    baseline_relative_l2 = _relative_l2(reference, output_baseline)
    residual_fused.zero_()
    fused_once()
    torch.cuda.synchronize()
    fused_relative_l2 = _relative_l2(reference, output_fused)
    torch.testing.assert_close(output_fused, output_baseline, rtol=2e-2, atol=2e-2)

    def measure_once(operation) -> float:
        if operation is baseline_once:
            residual_baseline.zero_()
        else:
            residual_fused.zero_()
        return _measure(operation, warmup=args.warmup, repeat=args.repeat)

    baseline_ms, fused_ms = measure_balanced(measure_once, baseline_once, fused_once)

    allocated_before = torch.cuda.memory_allocated(device)
    fused_once()
    torch.cuda.synchronize(device)
    steady_allocation_bytes = torch.cuda.memory_allocated(device) - allocated_before

    if rank == 0:
        result = {
            "benchmark": "fused_dequant_executor_end_to_end",
            "world_size": world_size,
            "bucket_mib": args.bucket_mib,
            "numel": numel,
            "dtype": args.dtype,
            "bit": config.bit,
            "group_size": config.group_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "measurement_order": "baseline-fused-fused-baseline",
            "baseline_ms": baseline_ms,
            "fused_ms": fused_ms,
            "speedup": baseline_ms / fused_ms,
            "latency_reduction_percent": (baseline_ms - fused_ms) / baseline_ms * 100.0,
            "baseline_relative_l2": baseline_relative_l2,
            "fused_relative_l2": fused_relative_l2,
            "steady_allocation_bytes": steady_allocation_bytes,
            "fast_path": executor.last_execution_info.fast_path,
            "fallback_used": executor.last_execution_info.fallback_used,
            "fallback_reason": executor.last_execution_info.fallback_reason,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
