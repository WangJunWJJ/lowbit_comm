from __future__ import annotations

from types import MappingProxyType

import pytest

from ccdl_comm.optim import (
    AdamWShardUpdateRule,
    SgdShardUpdateRule,
    ShardedOptimizerConsumer,
    UpdatedParameterShard,
)
from ccdl_comm.shard import ReducedShard
from ccdl_comm.shard_layout import FlatParameterSlice, FlatShardLayout

torch = pytest.importorskip("torch")


def layout(*, rank: int = 1) -> FlatShardLayout:
    return FlatShardLayout(
        original_numel=5,
        padded_numel=6,
        shard_numel=3,
        world_size=2,
        shard_index=rank,
        parameters=(
            FlatParameterSlice(
                index=0,
                offset=0,
                numel=5,
                shape=(5,),
                dtype="fp32",
                requires_grad=True,
            ),
        ),
    )


def reduced_shard(
    values: tuple[float, float, float] = (1.0, 2.0, 99.0),
    *,
    rank: int = 1,
) -> ReducedShard:
    return ReducedShard(
        shard=torch.tensor(values),
        shard_index=rank,
        shard_numel=3,
        original_shape=(5,),
        original_numel=5,
        padded_numel=6,
        world_size=2,
        reduce="mean",
        dtype="fp32",
    )


def test_consumer_updates_only_valid_values_and_returns_layout_version() -> None:
    parameter = torch.tensor([4.0, 5.0, 0.0])
    consumer = ShardedOptimizerConsumer(
        layout=layout(),
        parameter_shard=parameter,
        update_rule=SgdShardUpdateRule(learning_rate=0.1),
        layout_version=7,
    )

    updated = consumer.consume(reduced_shard(), step=1)

    assert isinstance(updated, UpdatedParameterShard)
    assert updated.shard is parameter
    assert updated.layout_version == 7
    assert updated.valid_numel == 2
    assert updated.metadata == {"update_rule": "sgd", "step": 1}
    assert isinstance(updated.metadata, MappingProxyType)
    torch.testing.assert_close(parameter, torch.tensor([3.9, 4.8, 0.0]))


def test_layout_mismatch_does_not_mutate_parameter_shard() -> None:
    parameter = torch.tensor([1.0, 2.0, 3.0])
    consumer = ShardedOptimizerConsumer(
        layout=layout(rank=0),
        parameter_shard=parameter,
        update_rule=SgdShardUpdateRule(learning_rate=0.1),
    )
    before = parameter.clone()

    with pytest.raises(ValueError, match="does not match optimizer layout"):
        consumer.consume(reduced_shard(rank=1), step=1)

    torch.testing.assert_close(parameter, before)


@pytest.mark.parametrize("learning_rate", (True, 0.0, -1.0, float("nan"), float("inf")))
def test_sgd_rule_rejects_invalid_learning_rate(learning_rate: object) -> None:
    with pytest.raises((TypeError, ValueError), match="finite positive number"):
        SgdShardUpdateRule(learning_rate=learning_rate)


@pytest.mark.parametrize("step", (True, 0, -1, 1.5))
def test_consumer_rejects_invalid_step_without_mutation(step: object) -> None:
    parameter = torch.tensor([4.0, 5.0, 0.0])
    consumer = ShardedOptimizerConsumer(
        layout=layout(),
        parameter_shard=parameter,
        update_rule=SgdShardUpdateRule(learning_rate=0.1),
    )
    before = parameter.clone()

    with pytest.raises((TypeError, ValueError), match="step must be a positive integer"):
        consumer.consume(reduced_shard(), step=step)

    torch.testing.assert_close(parameter, before)


def test_consumer_rejects_parameter_workspace_shape_before_update() -> None:
    parameter = torch.tensor([4.0, 5.0])

    with pytest.raises(ValueError, match="parameter shard numel"):
        ShardedOptimizerConsumer(
            layout=layout(),
            parameter_shard=parameter,
            update_rule=SgdShardUpdateRule(learning_rate=0.1),
        )


def test_updated_parameter_shard_rejects_inconsistent_valid_numel() -> None:
    with pytest.raises(ValueError, match="valid_numel"):
        UpdatedParameterShard(
            shard=torch.zeros(3),
            shard_index=1,
            shard_numel=3,
            valid_numel=4,
            original_numel=5,
            padded_numel=6,
            world_size=2,
            dtype="fp32",
            layout_version=0,
        )


def test_adamw_rule_matches_torch_for_rank_local_values() -> None:
    initial = torch.tensor([4.0, 5.0, 0.0])
    parameter = initial.clone()
    reference = torch.nn.Parameter(initial[:2].clone())
    reference_optimizer = torch.optim.AdamW(
        [reference],
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1.0e-6,
        weight_decay=0.1,
    )
    state: dict[str, object] = {}
    consumer = ShardedOptimizerConsumer(
        layout=layout(),
        parameter_shard=parameter,
        update_rule=AdamWShardUpdateRule(
            learning_rate=0.01,
            betas=(0.8, 0.9),
            epsilon=1.0e-6,
            weight_decay=0.1,
        ),
        state=state,
    )

    for step, gradient in enumerate(((1.0, 2.0, 99.0), (-3.0, 4.0, 99.0)), start=1):
        reference.grad = torch.tensor(gradient[:2])
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        consumer.consume(reduced_shard(gradient), step=step)

    torch.testing.assert_close(parameter[:2], reference.detach(), rtol=1.0e-6, atol=1.0e-7)
    assert parameter[2].item() == 0.0
    assert state["step"] == 2
    torch.testing.assert_close(state["exp_avg"][2:], torch.zeros(1))
    torch.testing.assert_close(state["exp_avg_sq"][2:], torch.zeros(1))


def test_adamw_rule_rejects_missing_mutable_state_without_mutation() -> None:
    parameter = torch.tensor([4.0, 5.0, 0.0])
    consumer = ShardedOptimizerConsumer(
        layout=layout(),
        parameter_shard=parameter,
        update_rule=AdamWShardUpdateRule(learning_rate=0.01),
    )
    before = parameter.clone()

    with pytest.raises(TypeError, match="mutable mapping"):
        consumer.consume(reduced_shard(), step=1)

    torch.testing.assert_close(parameter, before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"betas": (1.0, 0.9)}, "betas"),
        ({"betas": (0.9, -0.1)}, "betas"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"weight_decay": -0.1}, "weight_decay"),
    ),
)
def test_adamw_rule_rejects_invalid_hyperparameters(
    kwargs: dict[str, object],
    message: str,
) -> None:
    arguments = {"learning_rate": 0.01, **kwargs}
    with pytest.raises((TypeError, ValueError), match=message):
        AdamWShardUpdateRule(**arguments)
