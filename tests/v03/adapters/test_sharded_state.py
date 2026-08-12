from __future__ import annotations

import pytest

from lowbit_comm.adapters.sharded.consumer import ShardedMasterState
from lowbit_comm.adapters.sharded.qwd import (
    SafeInt8QWDPolicy,
    apply_quantized_weight_difference,
    full_precision_refresh,
    prepare_parameter_delta,
)
from lowbit_comm.core import DataType, ReducedShardValue


torch = pytest.importorskip("torch")


def _gradient() -> ReducedShardValue:
    return ReducedShardValue(
        tensor=torch.tensor([0.5, -0.25], dtype=torch.float16),
        shard_index=1,
        shard_numel=2,
        original_shape=(3,),
        original_numel=3,
        world_size=2,
        reduction="mean",
        dtype=DataType.FP16,
        layout_version=7,
    )


def test_master_state_consumes_only_valid_reduced_shard_values() -> None:
    state = ShardedMasterState(
        master=torch.tensor([2.0, 0.0], dtype=torch.float32),
        layout_version=7,
        world_size=2,
        rank=1,
    )

    state.sgd_step(_gradient(), learning_rate=0.1)

    assert torch.equal(state.master, torch.tensor([1.95, 0.0]))
    assert state.step == 1


def test_parameter_delta_contains_previous_lossy_writeback_error() -> None:
    master = torch.tensor([1.0, -2.0], dtype=torch.float32)
    model = torch.tensor([0.875, -1.75], dtype=torch.float16)

    delta = prepare_parameter_delta(master, model, valid_numel=2)
    model.add_(torch.tensor([0.10, -0.20], dtype=torch.float16))
    next_delta = prepare_parameter_delta(master, model, valid_numel=2)

    assert torch.allclose(delta, torch.tensor([0.125, -0.25]))
    assert torch.allclose(next_delta, master - model.float())


def test_qwd_adds_but_full_precision_refresh_overwrites_model_copy() -> None:
    model = torch.tensor([1.0, 2.0], dtype=torch.float16)

    apply_quantized_weight_difference(
        model,
        torch.tensor([0.25, -0.5], dtype=torch.float32),
        valid_numel=2,
    )
    assert torch.equal(model, torch.tensor([1.25, 1.5], dtype=torch.float16))

    full_precision_refresh(
        model,
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        valid_numel=2,
    )
    assert torch.equal(model, torch.tensor([3.0, 4.0], dtype=torch.float16))


def test_safe_policy_uses_warmup_periodic_error_and_capability_refresh() -> None:
    policy = SafeInt8QWDPolicy(warmup_steps=2, refresh_interval=4)

    assert policy.decide(step=1, relative_error=None, capability=True).reason == "warmup"
    assert policy.decide(step=3, relative_error=None, capability=True).mode == "qwd"
    assert policy.decide(step=4, relative_error=None, capability=True).reason == "periodic_refresh"
    assert policy.decide(step=5, relative_error=0.02, capability=True).reason == "error_threshold"
    assert policy.decide(step=5, relative_error=None, capability=False).reason == "capability"


def test_checkpoint_restore_validates_layout_and_forces_fp_refresh() -> None:
    original = ShardedMasterState(
        master=torch.tensor([3.0, 4.0], dtype=torch.float32),
        layout_version=5,
        world_size=2,
        rank=0,
    )
    checkpoint = original.state_dict()
    restored = ShardedMasterState(
        master=torch.zeros(2, dtype=torch.float32),
        layout_version=5,
        world_size=2,
        rank=0,
    )

    restored.load_state_dict(checkpoint)

    assert torch.equal(restored.master, original.master)
    assert restored.requires_fp_refresh is True
    with pytest.raises(ValueError, match="layout_version"):
        ShardedMasterState(
            master=torch.zeros(2, dtype=torch.float32),
            layout_version=6,
            world_size=2,
            rank=0,
        ).load_state_dict(checkpoint)


def test_adamw_master_update_matches_torch_reference() -> None:
    initial = torch.tensor([1.0, -2.0], dtype=torch.float32)
    gradient = _gradient()
    state = ShardedMasterState(
        master=initial.clone(),
        layout_version=7,
        world_size=2,
        rank=1,
    )
    reference = torch.nn.Parameter(initial[:1].clone())
    optimizer = torch.optim.AdamW([reference], lr=0.01)

    state.adamw_step(gradient, learning_rate=0.01)
    reference.grad = torch.tensor([0.5])
    optimizer.step()

    assert torch.allclose(state.master[:1], reference.detach(), atol=1e-7, rtol=1e-6)
    assert state.master[1].item() == initial[1].item()
