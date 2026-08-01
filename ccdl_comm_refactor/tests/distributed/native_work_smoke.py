"""Validate native c10d Work ownership on a real NCCL process group."""

from __future__ import annotations

import json
import os


def main() -> None:
    """Run one asynchronous all-reduce through the native Work wrapper."""

    import torch
    import torch.distributed as dist

    from ccdl_comm.cuda.loader import load_cuda_extension

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    extension_status = load_cuda_extension()
    if not extension_status.available or extension_status.module is None:
        raise RuntimeError(extension_status.reason or "CUDA extension is unavailable")
    extension = extension_status.module
    tensor = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    transport_work = dist.all_reduce(tensor, async_op=True)

    class WrappedWork:
        def __init__(self, handle):
            self.handle = handle

        def wait(self):
            raise AssertionError("native wrapper called Python wait()")

    work = extension.CompressedWork(
        tensor,
        WrappedWork(transport_work),
        None,
        [tensor],
    )

    if not work.uses_native_transport:
        raise AssertionError("torch.distributed.Work did not bind to native c10d::Work")
    if not isinstance(work.query(), bool):
        raise AssertionError("query() must return bool")
    value = float(work.wait().item())
    expected = world_size * (world_size + 1) / 2
    if value != expected:
        raise AssertionError(f"all-reduce result {value} != {expected}")

    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "uses_native_transport": work.uses_native_transport,
                    "result": value,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
