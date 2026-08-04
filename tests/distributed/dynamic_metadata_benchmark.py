"""Benchmark object and device-tensor metadata for dynamic all-gather."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from ccdl_comm import CompressionConfig, compile_dynamic_all_gather
from ccdl_comm.quantization.sizing import estimate_quantized_size


def parse_args() -> argparse.Namespace:
    """Parse the reproducible Task 20 benchmark matrix."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--payload-kib",
        type=int,
        nargs="+",
        default=(1, 1024, 16384),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    return parser.parse_args()


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _numel_for_payload(target_bytes: int, *, group_size: int) -> int:
    bytes_per_group = group_size + 2
    groups = max(1, round(target_bytes / bytes_per_group))
    return groups * group_size


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    denominator = reference.float().norm().clamp_min(1e-12)
    return float((reference.float() - candidate.float()).norm() / denominator)


def _measure(
    executor,
    tensor: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        executor.run(tensor).wait()
    torch.cuda.synchronize(tensor.device)
    torch.cuda.reset_peak_memory_stats(tensor.device)
    cpu_run_us: list[float] = []
    cpu_total_us: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        started = time.perf_counter_ns()
        work = executor.run(tensor)
        submitted = time.perf_counter_ns()
        work.wait()
        end_event.record()
        torch.cuda.synchronize(tensor.device)
        completed = time.perf_counter_ns()
        cpu_run_us.append((submitted - started) / 1000.0)
        cpu_total_us.append((completed - started) / 1000.0)
        gpu_ms.append(float(start_event.elapsed_time(end_event)))
    return {
        "repeat": repeat,
        "cpu_run_p50_us": statistics.median(cpu_run_us),
        "cpu_run_p95_us": _percentile(cpu_run_us, 0.95),
        "cpu_total_p50_us": statistics.median(cpu_total_us),
        "cpu_total_p95_us": _percentile(cpu_total_us, 0.95),
        "gpu_p50_ms": statistics.median(gpu_ms),
        "gpu_p95_ms": _percentile(gpu_ms, 0.95),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(tensor.device),
    }


def main() -> None:
    """Run both metadata protocols under identical distributed conditions."""

    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    config = CompressionConfig(bit=args.bit, group_size=args.group_size)
    cases = []
    for payload_kib in args.payload_kib:
        target_bytes = payload_kib * 1024
        numel = _numel_for_payload(
            target_bytes,
            group_size=args.group_size,
        )
        shape_class = (numel + (world_size - 1) * args.group_size,)
        local_numel = numel + rank * args.group_size
        torch.manual_seed(20260803 + rank)
        tensor = torch.randn(local_numel, device=device, dtype=dtype)
        estimate = estimate_quantized_size(
            local_numel,
            dtype=args.dtype,
            config=config,
        )
        protocol_results = {}
        accuracy = {}
        for protocol in ("object_v1", "tensor_v1"):
            executor = compile_dynamic_all_gather(
                shape_class=shape_class,
                config=config,
                dtype=args.dtype,
                metadata_protocol=protocol,
            )
            gathered = executor.run(tensor).wait()
            reference_inputs = []
            for source_rank in range(world_size):
                torch.manual_seed(20260803 + source_rank)
                reference_inputs.append(
                    torch.randn(
                        numel + source_rank * args.group_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            accuracy[protocol] = max(
                _relative_l2(reference, candidate)
                for reference, candidate in zip(
                    reference_inputs,
                    gathered,
                    strict=True,
                )
            )
            protocol_results[protocol] = _measure(
                executor,
                tensor,
                warmup=args.warmup,
                repeat=args.repeat,
            )
        object_p50 = float(protocol_results["object_v1"]["cpu_total_p50_us"])
        tensor_p50 = float(protocol_results["tensor_v1"]["cpu_total_p50_us"])
        cases.append(
            {
                "target_payload_bytes": target_bytes,
                "rank0_numel": numel,
                "rank_local_payload_bytes": estimate.quantized_bytes,
                "protocols": protocol_results,
                "max_relative_l2": accuracy,
                "tensor_over_object_speedup": object_p50 / tensor_p50,
            }
        )
        del tensor
        torch.cuda.empty_cache()
    gathered_cases: list[object | None] = [None] * world_size
    dist.all_gather_object(gathered_cases, cases)
    if rank == 0:
        result = {
            "benchmark": "task20_dynamic_metadata",
            "world_size": world_size,
            "dtype": args.dtype,
            "bit": args.bit,
            "group_size": args.group_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rank_results": gathered_cases,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
