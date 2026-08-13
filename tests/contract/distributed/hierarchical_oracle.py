from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.backends.cuda.transports.hierarchical import (
    HierarchyBindings,
    compile_hierarchy,
    execute_hierarchical_mean,
)
from lowbit_comm.compiler.passes.topology import parse_topology_signature


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    signature = os.environ.get(
        "LOWBIT_TOPOLOGY",
        "node_ids=" + ",".join("0" for _ in range(world_size)),
    )
    topology = parse_topology_signature(signature, world_size=world_size)
    groups = {ranks: dist.new_group(list(ranks)) for ranks in topology.node_groups}
    leader_group = dist.new_group(list(topology.leaders))
    plan = compile_hierarchy(topology, rank=rank)
    tensor = torch.full((4096,), rank + 1.0, device="cuda", dtype=torch.float32)
    execute_hierarchical_mean(
        tensor,
        plan,
        HierarchyBindings(groups[plan.local_group], leader_group),
        world_size=world_size,
        dist=dist,
    )
    expected = (world_size + 1.0) / 2.0
    error = float((tensor - expected).abs().max())
    assert error == 0.0, error
    if rank == 0:
        print(f"HIERARCHY_ORACLE_OK world_size={world_size} max_abs_error={error}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
