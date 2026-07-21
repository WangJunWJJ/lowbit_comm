import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .logging_utils import JsonlLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    import ccdl.comm as comm
    from ccdl.comm import qall_reduce
    from ccdl.quantization import Quantizer

    comm.init()
    rank = dist.get_rank()
    logger = JsonlLogger(args.output, rank)
    torch.manual_seed(7000 + rank)
    for bit, topk, limit in ((8, 0, 0.02), (8, 2, 0.02), (4, 0, 0.25), (4, 2, 0.25)):
        quantizer = Quantizer(64, -1, bit, topk, False, "fp32")
        x = torch.randn(1 << 20, device="cuda")
        q = quantizer.quantize(x)
        y = quantizer.dequantize(q, x.shape)
        q_error = float((y - x).norm() / x.norm())
        reference = x.clone()
        dist.all_reduce(reference)
        actual = x.clone()
        qall_reduce(actual, op="sum", quantizer=quantizer, method="tree", keep_self=False)
        error = float((actual - reference).norm() / reference.norm())
        if not torch.isfinite(actual).all() or error > limit:
            raise AssertionError(f"bit={bit} topk={topk} relative L2 {error} > {limit}")
        logger.emit("preflight", bit=bit, topk=topk, q_bytes=q.numel(), expected_q_bytes=quantizer.get_lenq(x.shape), qdq_rel_l2=q_error, allreduce_rel_l2=error)
    dist.barrier()


if __name__ == "__main__":
    main()
