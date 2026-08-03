from __future__ import annotations

import pytest

from ccdl_comm.collectives.reduce_scatter import ReducedShard as LegacyReducedShard
from ccdl_comm.shard import ReducedShard


def test_legacy_reduced_shard_is_core_reexport() -> None:
    assert LegacyReducedShard is ReducedShard


def test_reduced_shard_preserves_logical_padding_semantics() -> None:
    shard = ReducedShard(
        shard="payload",
        shard_index=1,
        shard_numel=2,
        original_shape=(3,),
        original_numel=3,
        world_size=2,
        reduce="mean",
        metadata={"bucket": 7},
    )

    assert shard.logical_range == (2, 3)
    assert shard.valid_numel == 1
    assert shard.padding_numel == 1
    assert shard.to_metadata()["metadata"] == {"bucket": 7}
    with pytest.raises(TypeError):
        shard.metadata["bucket"] = 8  # type: ignore[index]


@pytest.mark.parametrize(
    "overrides",
    (
        {"padded_numel": 5},
        {"shard_numel": 3, "padded_numel": 4},
    ),
)
def test_reduced_shard_rejects_inconsistent_partition_metadata(overrides) -> None:
    arguments = {
        "shard": "payload",
        "shard_index": 0,
        "shard_numel": 2,
        "original_shape": (3,),
        "original_numel": 3,
        "world_size": 2,
        "reduce": "mean",
        "padded_numel": 4,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=r"shard_numel \* world_size"):
        ReducedShard(**arguments)
