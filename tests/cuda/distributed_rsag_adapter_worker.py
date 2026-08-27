"""Rank-local real-CUDA smoke for the installable RSAG/qWD adapter."""

# ruff: noqa: E402 -- direct torchrun scripts must add the repository root.

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from lowbit_comm.experimental import (
    RSAG_CUDA_EXTENSION_ABI,
    RSAG_EVIDENCE_SCHEMA_VERSION,
    RSAG_LOWBIT_COMM_VERSION,
    RSAGEnvironment,
    RSAGEvidence,
    RSAGQWDAdapter,
    detect_rsag_environment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=4097)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logical_bytes = args.numel * 2
    os.environ.setdefault(
        "LOWBIT_COMM_RSAG_ATTESTED_TOPOLOGY",
        "single_node_pcie",
    )
    os.environ.setdefault(
        "LOWBIT_COMM_RSAG_ATTESTED_TRANSPORT",
        "nccl_p2p",
    )
    environment = detect_rsag_environment(
        dist.group.WORLD,
        logical_bytes=logical_bytes,
        topology_class="single_node_pcie",
        transport="nccl_p2p",
    )
    identity = {
        "world_size": environment.world_size,
        "node_count": environment.node_count,
        "topology_class": environment.topology_class,
        "transport": environment.transport,
        "gpu_model": environment.gpu_model,
        "torch_version": environment.torch_version,
        "cuda_version": environment.cuda_version,
        "nccl_version": environment.nccl_version,
        "lowbit_comm_version": RSAG_LOWBIT_COMM_VERSION,
        "cuda_extension_abi": RSAG_CUDA_EXTENSION_ABI,
        "checkpoint_schema_version": environment.checkpoint_schema_version,
        "build_fingerprint": environment.build_fingerprint,
    }
    assert type(environment) is RSAGEnvironment
    evidence = RSAGEvidence(
        schema_version=RSAG_EVIDENCE_SCHEMA_VERSION,
        min_logical_bytes=logical_bytes,
        max_logical_bytes=logical_bytes,
        seed_speedups_percent=(0.01, 0.02, 0.03),
        quality_passed=True,
        **identity,
    )
    adapter = RSAGQWDAdapter(environment, (evidence,))
    plans = adapter.create_plans(
        dist.group.WORLD,
        global_numel=args.numel,
        rank=rank,
    )

    gradient = (
        torch.linspace(
            -1.0,
            1.0,
            args.numel,
            dtype=torch.float32,
            device="cuda",
        )
        .add_(rank * 0.125)
        .to(torch.float16)
    )
    reduced = plans.gradient_plan.execute(gradient).wait()
    assert reduced.value.dtype is torch.float16
    assert reduced.value.shape == (plans.layout.padded_numel,)
    assert torch.isfinite(reduced.value).all()

    model = torch.zeros(
        plans.layout.padded_numel * world_size,
        dtype=torch.float16,
        device="cuda",
    )
    model[: args.numel].copy_(gradient)
    master = torch.zeros(
        plans.layout.padded_numel,
        dtype=torch.float32,
        device="cuda",
    )
    if plans.layout.valid_numel:
        master[: plans.layout.valid_numel].copy_(
            model.narrow(
                0,
                plans.layout.start,
                plans.layout.valid_numel,
            )
        )
    qwd_result = plans.qwd_plan.execute(
        master,
        model,
        adapter.mode(1),
    ).wait()
    assert qwd_result.dtype is torch.float16
    assert qwd_result.shape == model.shape
    assert torch.isfinite(qwd_result).all()

    refresh_result = plans.qwd_plan.execute(
        master,
        model,
        adapter.mode(0),
    ).wait()
    assert refresh_result.dtype is torch.float16
    assert refresh_result.shape == model.shape
    assert torch.isfinite(refresh_result).all()
    dist.barrier()
    if rank == 0:
        print(
            "RSAG_ADAPTER_OK "
            f"ranks={world_size} numel={args.numel} "
            f"nccl={identity['nccl_version']}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
