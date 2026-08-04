"""A6000 multi-GPU lifecycle smoke for compiled reduced-shard workspaces."""

from __future__ import annotations

import argparse
import json
import os
import time

def expected_workspace_count(world_size: int) -> int:
    """Return one internal send and recv workspace per peer."""

    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return 2 * world_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=8_388_608)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--async-op", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cached-bytes", type=int, default=134_217_728)
    return parser.parse_args()


def main() -> None:
    import torch
    import torch.distributed as dist

    from ccdl_comm import (
        CommunicationPlan,
        CompileContext,
        CompressionConfig,
        WorkspacePolicy,
        compile,
    )
    from ccdl_comm.cuda.backend import register_cuda_backends
    from ccdl_comm.cuda.loader import load_cuda_extension
    from ccdl_comm.registry import BackendRegistry

    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260801 + rank)
    source = torch.randn(args.numel, dtype=torch.float16, device=device)

    shard_numel = (args.numel + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    if padded_numel == args.numel:
        reference_full = source.clone()
    else:
        reference_full = torch.cat(
            (source, source.new_zeros((padded_numel - args.numel,))),
            dim=0,
        )
    dist.all_reduce(reference_full)
    reference_full /= world_size
    reference = reference_full.narrow(0, rank * shard_numel, shard_numel)

    extension = load_cuda_extension()
    if not extension.available:
        raise RuntimeError(extension.reason)
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=extension)
    compiled = compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8, group_size=64),
            output_layout="shard",
            async_op=args.async_op,
            workspace_policy=WorkspacePolicy(
                cache=True,
                max_cached_bytes=args.max_cached_bytes,
            ),
        ),
        CompileContext(
            rank=rank,
            world_size=world_size,
            device=str(device),
            shape=tuple(source.shape),
            dtype="fp16",
        ),
        registry=registry,
    )
    pool = compiled.executor.workspace_pool
    if pool is None:
        raise AssertionError("compiled reduced-shard executor did not retain its workspace pool")

    def run_once() -> torch.Tensor:
        return compiled.run(source).wait().shard

    retained = run_once()
    retained_snapshot = retained.clone()
    distinct = run_once()
    torch.cuda.synchronize(device)
    if retained.data_ptr() == distinct.data_ptr():
        raise AssertionError("returned reduced shard was reused while still retained")
    if not torch.equal(retained, retained_snapshot):
        raise AssertionError("retained reduced shard was overwritten by the next run")

    for _ in range(args.warmup):
        candidate = run_once()
    torch.cuda.synchronize(device)
    warmup_stats = pool.stats
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.repeat):
        candidate = run_once()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000 / args.repeat
    final_stats = pool.stats

    relative_l2 = float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )
    if relative_l2 > 0.02:
        raise AssertionError(f"relative L2 {relative_l2} exceeds 0.02")
    if final_stats.misses != warmup_stats.misses:
        raise AssertionError(
            "steady-state allocator miss count changed: "
            f"{warmup_stats.misses} -> {final_stats.misses}"
        )
    expected_hits = args.repeat * expected_workspace_count(world_size)
    if final_stats.hits - warmup_stats.hits != expected_hits:
        raise AssertionError(
            "unexpected steady-state workspace hits: "
            f"{final_stats.hits - warmup_stats.hits} != {expected_hits}"
        )
    if final_stats.in_flight_bytes != 0:
        raise AssertionError(
            f"workspace bytes remain in flight after synchronize: {final_stats.in_flight_bytes}"
        )

    latency = torch.tensor([elapsed_ms], dtype=torch.float64, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "async_op": args.async_op,
                    "numel": args.numel,
                    "latency_ms": float(latency.item()),
                    "relative_l2": relative_l2,
                    "warmup_stats": warmup_stats.__dict__,
                    "final_stats": final_stats.__dict__,
                    "steady_state_allocator_misses": final_stats.misses - warmup_stats.misses,
                    "steady_state_hits": final_stats.hits - warmup_stats.hits,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
