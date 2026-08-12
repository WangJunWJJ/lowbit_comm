from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lowbit_comm.adapters.sharded import (
    SafeInt8QWDPolicy,
    ShardedMasterState,
    full_precision_refresh,
    prepare_parameter_delta,
)
from lowbit_comm.backends.cuda import (
    dequantize_into,
    load_cuda_extension,
    payload_nbytes,
    quantize_into,
)
from lowbit_comm.core import DataType, QuantizedWire, ReducedShardValue


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    status = load_cuda_extension()
    if not status.available:
        raise RuntimeError(status.reason)
    shard_numel = 256
    original_numel = shard_numel * world_size
    initial = torch.linspace(
        -1.0,
        1.0,
        original_numel,
        device="cuda",
        dtype=torch.float32,
    )
    model = initial.to(torch.float16)
    master = initial.narrow(0, rank * shard_numel, shard_numel).clone()
    state = ShardedMasterState(
        master=master,
        layout_version=3,
        world_size=world_size,
        rank=rank,
    )
    reference = torch.nn.Parameter(initial.clone())
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01)
    policy = SafeInt8QWDPolicy(warmup_steps=2, refresh_interval=16)
    wire = QuantizedWire(8, 64)
    payload_size = payload_nbytes(shard_numel, dtype=DataType.FP32, wire=wire)
    payload = torch.empty(payload_size, device="cuda", dtype=torch.uint8)
    gathered_payload = torch.empty(
        world_size * payload_size,
        device="cuda",
        dtype=torch.uint8,
    )
    restored = torch.empty(shard_numel, device="cuda", dtype=torch.float32)
    gathered_master = torch.empty(original_numel, device="cuda", dtype=torch.float32)
    initial_loss = float(model.float().square().mean())
    qwd_steps = 0
    refresh_steps = 0
    for step in range(1, 101):
        gradient = model.float()
        local_gradient = gradient.narrow(0, rank * shard_numel, shard_numel)
        reduced = ReducedShardValue(
            tensor=local_gradient.to(torch.float16),
            shard_index=rank,
            shard_numel=shard_numel,
            original_shape=(original_numel,),
            original_numel=original_numel,
            world_size=world_size,
            reduction="mean",
            dtype=DataType.FP16,
            layout_version=3,
        )
        state.adamw_step(reduced, learning_rate=0.01)
        reference.grad = gradient
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        decision = policy.decide(step=step, relative_error=None, capability=True)
        if decision.mode == "fp_refresh":
            dist.all_gather_into_tensor(gathered_master, state.master)
            full_precision_refresh(model, gathered_master, valid_numel=original_numel)
            state.mark_fp_refreshed()
            refresh_steps += 1
        else:
            local_model = model.narrow(0, rank * shard_numel, shard_numel)
            delta = prepare_parameter_delta(
                state.master,
                local_model,
                valid_numel=shard_numel,
            )
            quantize_into(delta, payload, wire, extension_status=status)
            dist.all_gather_into_tensor(gathered_payload, payload)
            for source_rank in range(world_size):
                source = gathered_payload.narrow(
                    0,
                    source_rank * payload_size,
                    payload_size,
                )
                dequantize_into(
                    source,
                    restored,
                    wire,
                    dtype=DataType.FP32,
                    extension_status=status,
                )
                model.narrow(0, source_rank * shard_numel, shard_numel).add_(restored)
            qwd_steps += 1
    dist.all_gather_into_tensor(gathered_master, state.master)
    rank_min = model.clone()
    rank_max = model.clone()
    dist.all_reduce(rank_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
    rank_gap = float((rank_max - rank_min).abs().max())
    master_error = float((gathered_master - reference.detach()).abs().max())
    final_loss = float(model.float().square().mean())
    reference_loss = float(reference.detach().square().mean())
    relative_loss_gap = abs(final_loss - reference_loss) / max(reference_loss, 1.0e-12)
    assert rank_gap == 0.0, rank_gap
    assert master_error < 0.02, master_error
    assert final_loss < initial_loss * 0.2, (initial_loss, final_loss)
    assert relative_loss_gap < 0.05, relative_loss_gap
    if rank == 0:
        print(
            "QWD_ORACLE_OK "
            f"world_size={world_size} qwd_steps={qwd_steps} refresh_steps={refresh_steps} "
            f"rank_gap={rank_gap:.8f} master_error={master_error:.8f} "
            f"loss={final_loss:.8f} reference_loss={reference_loss:.8f} "
            f"relative_loss_gap={relative_loss_gap:.8f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
