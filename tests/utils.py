import argparse
import time
import torch

def do_bench(f, args = (), warmup = 3, repeat = 10):
    st = torch.Event("cuda", enable_timing=True)
    ed = torch.Event("cuda", enable_timing=True)
    a = torch.randn((1024,1024), device='cuda', dtype=torch.float32)
    for _ in range(3):
        a @ a
    for i in range(warmup):
        f(*args)
    st.record()
    for i in range(repeat):
        f(*args)
    ed.record()
    ed.synchronize()
    return st.elapsed_time(ed) * 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--size", type=str, default="4m")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--bit", type=int, default=8)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--stochastic", action='store_true')
    parser.add_argument("--keep-self", action='store_true')
    parser.add_argument("--dummy", action='store_true')
    parser.add_argument("--method", default=None)


    args = parser.parse_args()

    def _parse_size(s: str):
        s = s.lower()
        if s.endswith('m'):
            return int(s[:-1]) * 1024 * 1024
        elif s.endswith('k'):
            return int(s[:-1]) * 1024
        else:
            return int(s)

    args.size = _parse_size(args.size)

    STR_TO_TORCH_DTYPE = {
        "fp16": torch.half,
        "bf16": torch.bfloat16,
        "fp32": torch.float
    }

    args.torch_dtype = STR_TO_TORCH_DTYPE[args.dtype]

    return args

def print_args(args, world_size = None):
    print("Arguments:")
    if world_size is not None:
        print(" - World Size:", world_size)
    print(" - Size:", args.size)
    print(" - Group Size:", args.group_size)
    print(" - Dtype:", args.dtype)
    print(" - Dim:", args.dim)
    print(" - Bit:", args.bit)
    print(" - Topk:", args.topk)
    print(" - Stochastic:", args.stochastic)
    print(" - Keep Self:", args.keep_self)
    print(" - Dummy:", args.dummy)
    if args.method is not None:
        print(" - Method:", args.method)
    print("")
