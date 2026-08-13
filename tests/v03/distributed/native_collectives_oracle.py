from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda import CudaNativeCollectives


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    native = CudaNativeCollectives()
    value = torch.tensor([rank + 1.0], device="cuda")
    native.all_reduce(value, op=dist.ReduceOp.SUM).wait()
    assert value.item() == world_size * (world_size + 1) / 2
    gathered = torch.empty(world_size, device="cuda")
    native.all_gather_into_tensor(
        gathered, torch.tensor([rank], device="cuda", dtype=torch.float32)
    ).wait()
    assert torch.equal(gathered, torch.arange(world_size, device="cuda").float())
    rs_output = torch.empty(1, device="cuda")
    rs_input = torch.arange(world_size, device="cuda").float() + rank
    native.reduce_scatter_tensor(rs_output, rs_input, op=dist.ReduceOp.SUM).wait()
    assert rs_output.item() == rank * world_size + world_size * (world_size - 1) / 2
    a2a_output = torch.empty(world_size, device="cuda")
    a2a_input = torch.arange(world_size, device="cuda").float() + rank * 10
    native.all_to_all_single(a2a_output, a2a_input).wait()
    assert torch.equal(
        a2a_output,
        torch.tensor([source * 10 + rank for source in range(world_size)], device="cuda").float(),
    )
    broadcast = torch.tensor([7.0 if rank == 0 else 0.0], device="cuda")
    native.broadcast(broadcast, src=0).wait()
    assert broadcast.item() == 7.0
    reduced = torch.tensor([rank + 1.0], device="cuda")
    native.reduce(reduced, dst=0, op=dist.ReduceOp.SUM).wait()
    if rank == 0:
        assert reduced.item() == world_size * (world_size + 1) / 2
    gather_list = [torch.empty(1, device="cuda") for _ in range(world_size)] if rank == 0 else None
    native.gather(
        torch.tensor([rank], device="cuda", dtype=torch.float32),
        gather_list=gather_list,
        dst=0,
    ).wait()
    if rank == 0:
        assert [item.item() for item in gather_list] == list(range(world_size))
    scatter_list = (
        [
            torch.tensor([rank + 10.0], device="cuda", dtype=torch.float32)
            for rank in range(world_size)
        ]
        if rank == 0
        else None
    )
    scattered = torch.empty(1, device="cuda")
    native.scatter(scattered, scatter_list=scatter_list, src=0).wait()
    assert scattered.item() == rank + 10.0
    native.barrier().wait()
    if rank == 0:
        print(f"NATIVE_COLLECTIVES_OK world_size={world_size}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
