import argparse
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from .logging_utils import JsonlLogger


SIZES_MIB = (1, 4, 16, 64, 256)
VARIANTS = {
    "nccl_fp32": (None, None),
    "ccdl_int8_k0": (8, 0),
    "ccdl_int8_k2": (8, 2),
    "ccdl_int4_k0": (4, 0),
    "ccdl_int4_k2": (4, 2),
}


def _measure(operation, repeat):
    samples = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    import ccdl.comm as comm
    from ccdl.comm import qall_reduce
    from ccdl.quantization import Quantizer

    comm.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger = JsonlLogger(args.output, rank)
    for size_mib in SIZES_MIB:
        numel = size_mib * 2**20 // 4
        source = torch.randn(numel, device="cuda")
        for variant, (bit, topk) in VARIANTS.items():
            work = source.clone()
            if variant == "nccl_fp32":
                operation = lambda: dist.all_reduce(work, op=dist.ReduceOp.SUM)
                q_bytes = source.numel() * source.element_size()
                error = 0.0
            else:
                quantizer = Quantizer(64, -1, bit, topk, False, "fp32")
                operation = lambda: qall_reduce(work, op="sum", quantizer=quantizer, method="tree", keep_self=False)
                q_bytes = quantizer.get_lenq(source.shape)
                reference = source.clone()
                dist.all_reduce(reference)
                candidate = source.clone()
                qall_reduce(candidate, op="sum", quantizer=quantizer, method="tree", keep_self=False)
                error = float((candidate - reference).norm() / reference.norm())
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            samples = _measure(operation, args.repeat)
            ordered = sorted(samples)
            p50 = statistics.median(ordered)
            p95 = ordered[int(0.95 * (len(ordered) - 1))]
            logger.emit(
                "comm_summary",
                variant=variant,
                size_mib=size_mib,
                world_size=world_size,
                p50_ms=p50,
                p95_ms=p95,
                mean_ms=statistics.mean(samples),
                algorithm_gbps=(size_mib / 1024) / (p50 / 1000),
                bus_gbps=(size_mib / 1024) / (p50 / 1000) * 2 * (world_size - 1) / world_size,
                q_bytes=q_bytes,
                compression=(source.numel() * source.element_size()) / q_bytes,
                rel_l2=error,
                samples_ms=samples,
            )
        dist.barrier()


if __name__ == "__main__":
    main()
