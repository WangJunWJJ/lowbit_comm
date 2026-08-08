"""Two/four-rank qWD master-state and replicated-model correctness smoke."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any


def validate_qwd_configuration_packets(
    packets: Sequence[Mapping[str, object]],
) -> None:
    """Reject cross-rank qWD configuration differences before payload traffic."""

    active = tuple(packets)
    if not active:
        raise RuntimeError("qWD configuration packets must be non-empty")
    signature_keys = ("shape", "dtype", "payload_numel", "flags")
    reference = tuple(active[0].get(key) for key in signature_keys)
    for packet in active[1:]:
        signature = tuple(packet.get(key) for key in signature_keys)
        if signature != reference:
            raise RuntimeError("qWD configuration differs across ranks")


def validate_qwd_smoke_payload(payload: Mapping[str, object]) -> None:
    """Validate emitted evidence without importing torch or initializing NCCL."""

    world_size = payload.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise RuntimeError("qWD smoke world_size must be positive")
    losses = payload.get("losses")
    if not isinstance(losses, list) or not losses:
        raise RuntimeError("qWD smoke must report losses")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in losses
    ):
        raise FloatingPointError("qWD training losses must remain finite")
    rank_difference = payload.get("max_rank_parameter_difference")
    if not isinstance(rank_difference, (int, float)) or float(rank_difference) != 0.0:
        raise RuntimeError("qWD model copies differ across ranks")
    master_difference = payload.get("max_master_reference_difference")
    if (
        not isinstance(master_difference, (int, float))
        or not isfinite(float(master_difference))
        or float(master_difference) > 1.0e-6
    ):
        raise RuntimeError("FP32 master differs from AdamW reference")
    if payload.get("workspace_stable") is not True:
        raise RuntimeError("qWD steady-state workspace pointers changed")
    counts = payload.get("decision_counts")
    if not isinstance(counts, Mapping):
        raise RuntimeError("qWD smoke must report decision counts")
    if int(counts.get("qwd", 0)) < 1 or int(counts.get("fp_refresh", 0)) < 1:
        raise RuntimeError("qWD smoke must exercise qwd and fp_refresh")


def main() -> None:
    import torch
    import torch.distributed as dist

    from ccdl_comm.communication import (
        SafeInt8QWDPolicy,
        TorchQuantizedParameterDeltaRestore,
    )
    from ccdl_comm.config import CompressionConfig
    from ccdl_comm.cuda.loader import load_cuda_extension
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
    from examples.training.torch_sharded_adamw import TorchShardedAdamWStep

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(17)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 16),
        ).cuda().half()
        config = CompressionConfig(
            bit=8,
            group_size=64,
            compact=True,
            error_feedback=True,
        )
        policy = SafeInt8QWDPolicy(
            warmup_steps=0,
            refresh_interval=4,
            relative_error_threshold=10.0,
            error_check_interval=1,
        )
        extension = load_cuda_extension()
        if not extension.available:
            raise RuntimeError(extension.reason or "CUDA extension unavailable")
        restore = TorchQuantizedParameterDeltaRestore(
            config=config,
            model_dtype="fp16",
            extension_status=extension,
        )
        compiled_plan = None
        captured_reduced = None

        def reduce_scatter(flattened, *, out, layout):
            nonlocal compiled_plan, captured_reduced
            del layout
            if compiled_plan is None:
                compiled_plan = compile_cuda_shortcut(
                    flattened,
                    collective="reduce_scatter",
                    strategy="compressed",
                    output_layout="shard",
                    config=config,
                    async_op=False,
                    dtype="fp16",
                    extension_status=extension,
                )
                if compiled_plan.execution_info.fallback_used:
                    raise RuntimeError(
                        compiled_plan.execution_info.fallback_reason
                        or "compressed reduce-scatter used fallback"
                    )
            reduced = compiled_plan.run(flattened, out=out).wait()
            captured_reduced = reduced.shard.detach().float().clone()
            return reduced

        def global_error_ratio(residual_norm_sq, delta_norm_sq) -> float:
            norms = torch.stack(
                (residual_norm_sq.float(), delta_norm_sq.float())
            )
            dist.all_reduce(norms, op=dist.ReduceOp.SUM)
            return float((norms[0] / norms[1].clamp_min(1.0e-24)).sqrt())

        adapter = TorchShardedAdamWStep.from_parameters(
            model.parameters(),
            rank=rank,
            world_size=world_size,
            group_size=config.group_size,
            learning_rate=1.0e-3,
            reduce_scatter=reduce_scatter,
            restore=restore,
            weight_decay=1.0e-4,
            global_error_ratio=global_error_ratio,
            policy=policy,
        )
        _validate_configuration(
            adapter=adapter,
            policy=policy,
            config=config,
            world_size=world_size,
            torch=torch,
            dist=dist,
        )
        valid_numel = adapter.layout.valid_numel
        reference_parameter = torch.nn.Parameter(
            adapter.master_shard[:valid_numel].detach().clone()
        )
        reference_optimizer = torch.optim.AdamW(
            (reference_parameter,),
            lr=1.0e-3,
            weight_decay=1.0e-4,
        )
        losses: list[float] = []
        decision_counts = {"qwd": 0, "fp_refresh": 0}
        initial_pointers = None
        for step in range(1, 6):
            model.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cuda").manual_seed(
                1000 * rank + step
            )
            features = torch.randn(
                16,
                32,
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            targets = torch.randn(
                16,
                16,
                device="cuda",
                dtype=torch.float16,
                generator=generator,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(features), targets)
            loss.backward()
            metrics = adapter.step(step=step)
            if captured_reduced is None:
                raise RuntimeError("reduce-scatter did not expose its reduced shard")
            reference_optimizer.zero_grad(set_to_none=True)
            reference_parameter.grad = captured_reduced[:valid_numel].clone()
            reference_optimizer.step()
            decision_counts[metrics.parameter_communication_mode] += 1
            torch.cuda.synchronize()
            losses.append(float(loss))
            if initial_pointers is None:
                initial_pointers = {
                    "adapter": adapter.workspace_pointers(),
                    "restore": restore.workspace_pointers(),
                }

        final_pointers = {
            "adapter": adapter.workspace_pointers(),
            "restore": restore.workspace_pointers(),
        }
        local_master_difference = (
            adapter.master_shard[:valid_numel] - reference_parameter.detach()
        ).abs().max()
        dist.all_reduce(local_master_difference, op=dist.ReduceOp.MAX)
        flat_model = torch.cat(
            [parameter.detach().reshape(-1) for parameter in model.parameters()]
        )
        rank_zero_model = flat_model.clone()
        dist.broadcast(rank_zero_model, src=0)
        rank_difference = (flat_model - rank_zero_model).abs().max()
        dist.all_reduce(rank_difference, op=dist.ReduceOp.MAX)
        payload = {
            "world_size": world_size,
            "losses": losses,
            "max_rank_parameter_difference": float(rank_difference),
            "max_master_reference_difference": float(local_master_difference),
            "decision_counts": decision_counts,
            "workspace_stable": initial_pointers == final_pointers,
            "restore_fast_path": restore.last_fast_path,
            "fallback_reason": restore.last_fallback_reason,
        }
        validate_qwd_smoke_payload(payload)
        if restore.last_fallback_reason is not None:
            raise RuntimeError(restore.last_fallback_reason)
        if rank == 0:
            print(json.dumps(payload, sort_keys=True))
    finally:
        dist.destroy_process_group()


def _validate_configuration(
    *,
    adapter: Any,
    policy: Any,
    config: Any,
    world_size: int,
    torch: Any,
    dist: Any,
) -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_NUMEL,
        decode_metadata_packets,
        encode_metadata_packet,
    )

    shape = (
        0,
        world_size,
        config.bit,
        config.group_size,
        *policy.configuration_packet(),
    )
    packet = encode_metadata_packet(
        shape=shape,
        dtype=adapter.layout.dtype,
        payload_numel=adapter.layout.shard_numel,
        flags=1,
        torch=torch,
        device=adapter.master_shard.device,
    )
    gathered = torch.empty(
        METADATA_PACKET_NUMEL * world_size,
        dtype=torch.int64,
        device=adapter.master_shard.device,
    )
    dist.all_gather_into_tensor(gathered, packet)
    validate_qwd_configuration_packets(
        decode_metadata_packets(gathered, world_size=world_size)
    )


if __name__ == "__main__":
    main()
