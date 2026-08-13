from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from ccdl_comm.communication import TorchQuantizedParameterDeltaRestore
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.cuda.shortcut import compile_cuda_shortcut
from examples.training.torch_sharded_adamw import TorchShardedAdamWStep


def main() -> None:
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
        ).cuda()
        config = CompressionConfig(
            bit=8,
            group_size=64,
            compact=True,
            error_feedback=True,
        )
        extension = load_cuda_extension()
        if not extension.available:
            raise RuntimeError(extension.reason or "CUDA extension unavailable")
        compiled_plan = None

        def reduce_scatter(flattened, *, out, layout):
            nonlocal compiled_plan
            del layout
            if compiled_plan is None:
                compiled_plan = compile_cuda_shortcut(
                    flattened,
                    collective="reduce_scatter",
                    strategy="compressed",
                    output_layout="shard",
                    config=config,
                    async_op=False,
                    dtype="fp32",
                    extension_status=extension,
                )
                if compiled_plan.execution_info.fallback_used:
                    raise RuntimeError(
                        compiled_plan.execution_info.fallback_reason
                        or "compressed reduce-scatter used fallback"
                    )
            return compiled_plan.run(flattened, out=out).wait()

        def global_l2_norm(shard) -> float:
            squared = shard.float().square().sum()
            dist.all_reduce(squared, op=dist.ReduceOp.SUM)
            return float(squared.sqrt())

        restore = TorchQuantizedParameterDeltaRestore(
            config=config,
            model_dtype="fp32",
            extension_status=extension,
        )
        adapter = TorchShardedAdamWStep.from_parameters(
            model.parameters(),
            rank=rank,
            world_size=world_size,
            group_size=config.group_size,
            learning_rate=1.0e-3,
            reduce_scatter=reduce_scatter,
            restore=restore,
            weight_decay=1.0e-4,
            global_l2_norm=global_l2_norm,
        )
        pointers = adapter.workspace_pointers()
        losses = []
        for step in range(1, 4):
            model.zero_grad(set_to_none=True)
            generator = torch.Generator(device="cuda").manual_seed(1000 * rank + step)
            features = torch.randn(16, 32, device="cuda", generator=generator)
            targets = torch.randn(16, 16, device="cuda", generator=generator)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(features), targets)
            loss.backward()
            adapter.step(step=step, max_grad_norm=1.0)
            torch.cuda.synchronize()
            losses.append(float(loss))

        flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        rank_zero = flat.clone()
        dist.broadcast(rank_zero, src=0)
        max_difference = float((flat - rank_zero).abs().max())
        if not all(torch.isfinite(torch.tensor(losses))):
            raise FloatingPointError("training loss must remain finite")
        if max_difference != 0.0:
            raise RuntimeError(f"rank parameter difference is {max_difference}")
        if restore.last_fallback_reason is not None:
            raise RuntimeError(restore.last_fallback_reason)
        if adapter.workspace_pointers() != pointers:
            raise RuntimeError("adapter workspace pointers changed")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "world_size": world_size,
                        "losses": losses,
                        "max_rank_parameter_difference": max_difference,
                        "restore_fast_path": restore.last_fast_path,
                        "optimizer_state_numel": adapter.optimizer_state_numel,
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
