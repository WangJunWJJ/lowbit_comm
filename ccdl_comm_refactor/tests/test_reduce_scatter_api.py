import pytest

from ccdl_comm import CompressionConfig, ReducedShard, compressed_reduce_scatter, compressed_reduce_scatter_shard
from ccdl_comm.exceptions import UnsupportedCollective


class FakeTensor:
    shape = (4,)
    dtype = "float32"


def test_reduce_scatter_falls_back_when_transport_missing() -> None:
    calls = []

    def fallback(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.bit, op, async_op, dtype, extension_status))
        return "fallback-result"

    result = compressed_reduce_scatter(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        all_gather_fallback=fallback,
    )

    assert result == "fallback-result"
    assert calls[0][1:5] == (8, "mean", False, "auto")


def test_reduce_scatter_rejects_unsupported_op() -> None:
    with pytest.raises(UnsupportedCollective, match="reduce_scatter:max"):
        compressed_reduce_scatter(FakeTensor(), config=CompressionConfig(), op="max")


def test_reduce_scatter_shard_uses_injected_sharded_transport() -> None:
    calls = []

    def transport(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.bit, op, async_op, dtype, extension_status))
        return ReducedShard(
            shard="rank-local-shard",
            shard_index=1,
            shard_numel=2,
            original_shape=(4,),
            original_numel=4,
            world_size=2,
            reduce="mean",
        )

    result = compressed_reduce_scatter_shard(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        reduce_scatter_shard=transport,
        dtype="fp32",
    )

    assert result.shard == "rank-local-shard"
    assert result.shard_index == 1
    assert result.shard_numel == 2
    assert result.original_shape == (4,)
    assert result.original_numel == 4
    assert result.world_size == 2
    assert result.reduce == "mean"
    assert calls[0][1:5] == (8, "mean", False, "fp32")


def test_reduce_scatter_shard_requires_sharded_transport() -> None:
    with pytest.raises(UnsupportedCollective, match="reduce_scatter_shard:transport"):
        compressed_reduce_scatter_shard(FakeTensor(), config=CompressionConfig())
