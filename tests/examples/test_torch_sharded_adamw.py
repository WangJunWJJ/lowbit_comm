from __future__ import annotations

import pytest

from ccdl_comm.shard import ReducedShard
from examples.training.torch_sharded_adamw import TorchShardedAdamWStep

torch = pytest.importorskip("torch")


class ImmediateRestore:
    def restore(self, updated, *, out, async_op: bool):
        assert async_op is True
        out.copy_(updated.shard)
        return ImmediateResult(out)


class ImmediateResult:
    def __init__(self, value) -> None:
        self._value = value

    def wait(self):
        return self._value


def single_rank_reduce(flattened, *, out, layout):
    out.copy_(flattened)
    return ReducedShard(
        shard=out,
        shard_index=0,
        shard_numel=layout.shard_numel,
        original_shape=(layout.original_numel,),
        original_numel=layout.original_numel,
        padded_numel=layout.padded_numel,
        world_size=1,
        reduce="mean",
        dtype=layout.dtype,
    )


def test_arbitrary_module_step_matches_full_adamw_and_reuses_workspaces() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.GELU(),
        torch.nn.Linear(4, 2),
    )
    reference = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.GELU(),
        torch.nn.Linear(4, 2),
    )
    reference.load_state_dict(model.state_dict())
    reference_optimizer = torch.optim.AdamW(
        (
            {
                "params": [parameter for parameter in reference.parameters() if parameter.dim() >= 2],
                "weight_decay": 0.1,
            },
            {
                "params": [parameter for parameter in reference.parameters() if parameter.dim() < 2],
                "weight_decay": 0.0,
            },
        ),
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1.0e-6,
    )
    model_parameters = tuple(model.parameters())
    adapter = TorchShardedAdamWStep.from_parameters(
        model_parameters,
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        betas=(0.8, 0.9),
        epsilon=1.0e-6,
        weight_decay=0.1,
        weight_decays=tuple(
            0.1 if parameter.dim() >= 2 else 0.0 for parameter in model_parameters
        ),
        reduce_scatter=single_rank_reduce,
        restore=ImmediateRestore(),
    )
    pointers = adapter.workspace_pointers()
    assert all(
        actual is expected
        for actual, expected in zip(
            adapter.parameters,
            model.parameters(),
            strict=True,
        )
    )
    features = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    for step in range(1, 3):
        model.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(features), targets).backward()
        torch.nn.functional.mse_loss(reference(features), targets).backward()
        reference_optimizer.step()
        adapter.step(step=step)

    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert adapter.workspace_pointers() == pointers
    assert adapter.optimizer_state_numel == 2 * adapter.layout.shard_numel


def test_step_rejects_missing_gradient_before_parameter_update() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    adapter = TorchShardedAdamWStep.from_parameters(
        model.parameters(),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=ImmediateRestore(),
    )
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    model[0](torch.randn(2, 3)).sum().backward()

    with pytest.raises(RuntimeError, match="missing gradients"):
        adapter.step(step=1)

    for actual, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected)


def test_adapter_rejects_frozen_parameters_before_rebinding() -> None:
    trainable = torch.nn.Parameter(torch.tensor([1.0]))
    frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    with pytest.raises(ValueError, match="requires_grad"):
        TorchShardedAdamWStep.from_parameters(
            (trainable, frozen),
            rank=0,
            world_size=1,
            group_size=64,
            learning_rate=0.01,
            reduce_scatter=single_rank_reduce,
            restore=ImmediateRestore(),
        )


def test_step_clips_the_global_reduced_gradient_norm() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    adapter = TorchShardedAdamWStep.from_parameters(
        (parameter,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=ImmediateRestore(),
        global_l2_norm=lambda shard: float(torch.linalg.vector_norm(shard[:2])),
    )
    parameter.grad = torch.tensor([3.0, 4.0])

    metrics = adapter.step(step=1, max_grad_norm=1.0)

    assert metrics.gradient_norm == pytest.approx(5.0)
    assert metrics.clip_coefficient == pytest.approx(0.2)
