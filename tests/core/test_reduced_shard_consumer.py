from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ccdl_comm.consumer import ReducedShardConsumer
from ccdl_comm.shard import ReducedShard
from ccdl_comm.shard_layout import FlatParameterSlice, FlatShardLayout


class FakeTensor:
    def __init__(self, numel: int) -> None:
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class Consumer:
    def consume(self, reduced: ReducedShard) -> object:
        return reduced.shard


def parameter_slices(*, second_dtype: str = "fp16") -> tuple[FlatParameterSlice, ...]:
    return (
        FlatParameterSlice(0, 0, 4, (2, 2), "fp16", True),
        FlatParameterSlice(1, 4, 1, (1,), second_dtype, True),
    )


def layout(**overrides) -> FlatShardLayout:
    values = {
        "original_numel": 5,
        "padded_numel": 6,
        "shard_numel": 3,
        "world_size": 2,
        "shard_index": 1,
        "parameters": parameter_slices(),
    }
    values.update(overrides)
    return FlatShardLayout(**values)


def reduced_shard(**overrides) -> ReducedShard:
    values = {
        "shard": FakeTensor(3),
        "shard_index": 1,
        "shard_numel": 3,
        "original_shape": (5,),
        "original_numel": 5,
        "padded_numel": 6,
        "world_size": 2,
        "reduce": "mean",
        "dtype": "fp16",
    }
    values.update(overrides)
    return ReducedShard(**values)


def test_consumer_protocol_is_runtime_checkable() -> None:
    assert isinstance(Consumer(), ReducedShardConsumer)


def test_layout_validates_matching_reduced_shard() -> None:
    active_layout = layout()

    active_layout.validate_reduced_shard(reduced_shard())

    assert active_layout.shard_offset == 3
    assert active_layout.shard_end == 5
    assert active_layout.valid_numel == 2
    assert active_layout.padding_numel == 1
    assert active_layout.logical_range == (3, 5)


def test_layout_and_parameter_slices_are_immutable() -> None:
    active_layout = layout()

    with pytest.raises(FrozenInstanceError):
        active_layout.shard_index = 0
    with pytest.raises(FrozenInstanceError):
        active_layout.parameters[0].offset = 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_numel", True),
        ("padded_numel", 5.0),
        ("shard_numel", "3"),
        ("world_size", 0),
        ("shard_index", 2),
    ],
)
def test_layout_rejects_invalid_integer_metadata(field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        layout(**{field: value})


@pytest.mark.parametrize(
    "parameters",
    [
        (
            FlatParameterSlice(0, 0, 3, (3,), "fp16", True),
            FlatParameterSlice(1, 4, 1, (1,), "fp16", True),
        ),
        (
            FlatParameterSlice(0, 0, 4, (4,), "fp16", True),
            FlatParameterSlice(1, 3, 2, (2,), "fp16", True),
        ),
        (FlatParameterSlice(0, 0, 4, (4,), "fp16", True),),
    ],
)
def test_layout_requires_contiguous_complete_parameter_coverage(parameters) -> None:
    with pytest.raises(ValueError, match="contiguous.*original_numel"):
        layout(parameters=parameters)


def test_layout_rejects_mixed_parameter_dtype() -> None:
    with pytest.raises(ValueError, match="one dtype"):
        layout(parameters=parameter_slices(second_dtype="fp32"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shard_index": 0}, "shard_index"),
        ({"shard_numel": 4, "padded_numel": 8}, "shard_numel"),
        ({"original_shape": (6,), "original_numel": 6}, "original_numel"),
        ({"world_size": 3, "shard_numel": 2}, "world_size"),
        ({"dtype": "fp32"}, "dtype"),
        ({"shard": FakeTensor(2)}, "tensor numel"),
    ],
)
def test_layout_rejects_mismatched_reduced_shard(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        layout().validate_reduced_shard(reduced_shard(**overrides))
