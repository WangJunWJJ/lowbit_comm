import torch
import torch.distributed as dist
import ccdl.comm as comm
from ccdl.quantization import Quantizer


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.utils import do_bench, parse_args, print_args


comm.init()
torch.cuda.set_device(dist.get_rank())

world_size = dist.get_world_size()

args = parse_args()
if dist.get_rank() == 0:
    print_args(args, world_size)

dtype = args.torch_dtype
data = torch.randn((args.size,), device=torch.cuda.current_device(), dtype=dtype)

ground_truth = data.clone()
dist.all_reduce(ground_truth)
quantizer = Quantizer(args.group_size, args.dim, args.bit, args.topk, args.stochastic, args.dtype, dummy=args.dummy)


qret = data.clone()
hd = comm.qall_reduce(qret, quantizer=quantizer, op="sum", keep_self=False, method=args.method, async_op=True)
if hd is not None:
    hd.wait()

for rank in range(dist.get_world_size()):
    if rank == dist.get_rank():
        print("Rank", rank)
        a = ground_truth
        b = qret
        print(" - MAX_ERR:", f"{(a-b).abs().max().item(): .3f}")
    dist.barrier()

def q_func():
    hd = comm.qall_reduce(qret, quantizer=quantizer, keep_self=False, method=args.method, async_op=True)
    if hd is not None:
        hd.wait()

def func():
    dist.all_reduce(data)

a = do_bench(q_func)
b = do_bench(func)

if dist.get_rank() == 0:
    print(f"fp16: {b*1000}ms")
    print(f"quant: {a*1000}ms")
    print(f"Speed up: {(b/a- 1)*100:.2f}%")
