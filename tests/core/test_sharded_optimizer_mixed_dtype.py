from __future__ import annotations

from typing import Any

import pytest

from ccdl_comm.optim import ShardedOptimizerConsumer
from ccdl_comm.shard import ReducedShard
from ccdl_comm.shard_layout import FlatParameterSlice, FlatShardLayout


class FakeTensor:
    def __init__(self, values: tuple[float, ...], *, dtype: str) -> None:
        self.values = values
        self.dtype = dtype
        self.device = "cuda:0"

    def numel(self) -> int:
        return len(self.values)

    def is_contiguous(self) -> bool:
        return True


class RecordingUpdateRule:
    name = "recording"

    def __init__(self) -> None:
        self.gradient: Any = None

    def update(
        self,
        parameter_shard: Any,
        gradient_shard: Any,
        state: Any,
        *,
        valid_numel: int,
        step: int,
    ) -> Any:
        del state, valid_numel, step
        self.gradient = gradient_shard
        return parameter_shard


def test_consumer_transforms_low_precision_gradient_before_master_update() -> None:
    layout = FlatShardLayout(
        original_numel=3,
        padded_numel=3,
        shard_numel=3,
        world_size=1,
        shard_index=0,
        parameters=(
            FlatParameterSlice(
                index=0,
                offset=0,
                numel=3,
                shape=(3,),
                dtype="fp16",
                requires_grad=True,
            ),
        ),
    )
    master = FakeTensor((1.0, 2.0, 3.0), dtype="fp32")
    low_precision_gradient = FakeTensor((0.1, 0.2, 0.3), dtype="fp16")
    fp32_gradient = FakeTensor(low_precision_gradient.values, dtype="fp32")
    rule = RecordingUpdateRule()
    consumer = ShardedOptimizerConsumer(
        layout=layout,
        parameter_shard=master,
        update_rule=rule,
        gradient_transform=lambda gradient, parameter: fp32_gradient,
    )
    reduced = ReducedShard(
        shard=low_precision_gradient,
        shard_index=0,
        shard_numel=3,
        original_shape=(3,),
        original_numel=3,
        padded_numel=3,
        world_size=1,
        reduce="mean",
        dtype="fp16",
    )

    updated = consumer.consume(reduced, step=1)

    assert updated.shard is master
    assert rule.gradient is fp32_gradient


def test_consumer_rejects_noncallable_gradient_transform_at_construction() -> None:
    layout = _single_rank_layout()

    with pytest.raises(TypeError, match="gradient_transform must be callable"):
        ShardedOptimizerConsumer(
            layout=layout,
            parameter_shard=FakeTensor((1.0, 2.0, 3.0), dtype="fp32"),
            update_rule=RecordingUpdateRule(),
            gradient_transform=3,
        )


def test_consumer_rejects_transformed_gradient_with_wrong_numel() -> None:
    layout = _single_rank_layout()
    consumer = ShardedOptimizerConsumer(
        layout=layout,
        parameter_shard=FakeTensor((1.0, 2.0, 3.0), dtype="fp32"),
        update_rule=RecordingUpdateRule(),
        gradient_transform=lambda gradient, parameter: FakeTensor(
            (0.1, 0.2),
            dtype="fp32",
        ),
    )

    with pytest.raises(
        ValueError,
        match="transformed gradient shard must match layout shard_numel",
    ):
        consumer.consume(_reduced_gradient(), step=1)


def _single_rank_layout() -> FlatShardLayout:
    return FlatShardLayout(
        original_numel=3,
        padded_numel=3,
        shard_numel=3,
        world_size=1,
        shard_index=0,
        parameters=(
            FlatParameterSlice(
                index=0,
                offset=0,
                numel=3,
                shape=(3,),
                dtype="fp16",
                requires_grad=True,
            ),
        ),
    )


def _reduced_gradient() -> ReducedShard:
    return ReducedShard(
        shard=FakeTensor((0.1, 0.2, 0.3), dtype="fp16"),
        shard_index=0,
        shard_numel=3,
        original_shape=(3,),
        original_numel=3,
        padded_numel=3,
        world_size=1,
        reduce="mean",
        dtype="fp16",
    )
