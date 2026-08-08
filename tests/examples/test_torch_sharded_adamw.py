from __future__ import annotations

import pytest

from ccdl_comm.shard import ReducedShard
from examples.training.torch_sharded_adamw import (
    ShardedAdamWState,
    TorchShardedAdamWStep,
)

torch = pytest.importorskip("torch")


class ImmediateQWDRestore:
    def __init__(self) -> None:
        self.modes: list[str] = []

    def supports_qwd(self, updated, out) -> bool:
        del updated, out
        return True

    def restore_delta(self, delta, *, out, async_op: bool):
        assert async_op is True
        self.modes.append("qwd")
        out.add_(delta.shard.to(out.dtype))
        return ImmediateResult(out)

    def refresh(self, updated, *, out, async_op: bool):
        assert async_op is True
        self.modes.append("fp_refresh")
        out.copy_(updated.shard.to(out.dtype))
        return ImmediateResult(out)


class ImmediateResult:
    def __init__(self, value) -> None:
        self._value = value

    def wait(self):
        return self._value


class AlwaysQWDPolicy:
    error_check_interval = 1

    def decide(self, **kwargs):
        del kwargs
        from ccdl_comm.communication import ParameterCommunicationDecision

        return ParameterCommunicationDecision(mode="qwd", bit=8, reason="test")

    def configuration_packet(self):
        return (0, 512, 10_000_000, 1)


class RecordingLossyQWDRestore(ImmediateQWDRestore):
    def __init__(self) -> None:
        super().__init__()
        self.deltas = []

    def restore_delta(self, delta, *, out, async_op: bool):
        assert async_op is True
        self.modes.append("qwd")
        recorded = delta.shard.detach().clone()
        self.deltas.append(recorded)
        factor = 0.5 if len(self.deltas) == 1 else 1.0
        out.add_(recorded.to(out.dtype), alpha=factor)
        return ImmediateResult(out)


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
        restore=ImmediateQWDRestore(),
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
        restore=ImmediateQWDRestore(),
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
            restore=ImmediateQWDRestore(),
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
        restore=ImmediateQWDRestore(),
        global_l2_norm=lambda shard: float(torch.linalg.vector_norm(shard[:2])),
    )
    parameter.grad = torch.tensor([3.0, 4.0])

    metrics = adapter.step(step=1, max_grad_norm=1.0)

    assert metrics.gradient_norm == pytest.approx(5.0)
    assert metrics.clip_coefficient == pytest.approx(0.2)


def test_rank_local_adamw_state_round_trip_preserves_the_next_update() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    first_adapter = TorchShardedAdamWStep.from_parameters(
        (first,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=ImmediateQWDRestore(),
    )
    first.grad = torch.tensor([0.3, -0.2])
    first_adapter.step(step=1)
    saved_parameter = first.detach().clone()
    saved_state = first_adapter.export_adamw_state()
    assert isinstance(saved_state, ShardedAdamWState)
    assert saved_state.master_shard.dtype == torch.float32
    assert saved_state.exp_avg.dtype == torch.float32
    assert saved_state.exp_avg_sq.dtype == torch.float32

    second = torch.nn.Parameter(saved_parameter.clone())
    second_adapter = TorchShardedAdamWStep.from_parameters(
        (second,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=ImmediateQWDRestore(),
    )
    second_adapter.load_adamw_state(saved_state)
    first.grad = torch.tensor([-0.4, 0.1])
    second.grad = first.grad.clone()

    first_adapter.step(step=2)
    second_adapter.step(step=2)

    torch.testing.assert_close(second, first)


def test_adapter_keeps_fp32_master_behind_fp16_model_copy() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.float16)
    )
    restore = ImmediateQWDRestore()
    adapter = TorchShardedAdamWStep.from_parameters(
        (parameter,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=restore,
    )
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float16)

    adapter.step(step=1)

    assert adapter.master_shard.dtype == torch.float32
    assert parameter.dtype == torch.float16
    assert adapter.master_shard.data_ptr() != parameter.data_ptr()
    assert restore.modes == ["fp_refresh"]


def test_qwd_error_is_carried_by_next_master_minus_model_delta() -> None:
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, 2.0], dtype=torch.float16)
    )
    restore = RecordingLossyQWDRestore()
    adapter = TorchShardedAdamWStep.from_parameters(
        (parameter,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=restore,
        policy=AlwaysQWDPolicy(),
    )
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float16)
    adapter.step(step=1)
    first_master = adapter.master_shard[:2].detach().clone()
    first_model = parameter.detach().float().clone()
    first_unrestored = first_master - first_model

    parameter.grad = torch.zeros_like(parameter)
    adapter.step(step=2)
    second_master_update = adapter.master_shard[:2].detach() - first_master

    torch.testing.assert_close(
        restore.deltas[1][:2],
        first_unrestored + second_master_update,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_loading_checkpoint_forces_full_precision_refresh() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    first_restore = ImmediateQWDRestore()
    first_adapter = TorchShardedAdamWStep.from_parameters(
        (first,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=first_restore,
        policy=AlwaysQWDPolicy(),
    )
    first.grad = torch.tensor([0.2, -0.1])
    first_adapter.step(step=1)
    state = first_adapter.export_adamw_state()

    second = torch.nn.Parameter(torch.tensor([-9.0, -8.0]))
    second_restore = ImmediateQWDRestore()
    second_adapter = TorchShardedAdamWStep.from_parameters(
        (second,),
        rank=0,
        world_size=1,
        group_size=64,
        learning_rate=0.01,
        reduce_scatter=single_rank_reduce,
        restore=second_restore,
        policy=AlwaysQWDPolicy(),
    )
    second_adapter.load_adamw_state(state)
    second.grad = torch.zeros_like(second)

    second_adapter.step(step=2)

    assert second_restore.modes == ["fp_refresh"]
