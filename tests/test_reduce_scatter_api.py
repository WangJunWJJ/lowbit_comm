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


def test_reduce_scatter_shard_passes_caller_owned_output_to_injected_transport() -> None:
    output = object()
    calls = []

    def transport(tensor, *, config, op, async_op, dtype, extension_status, out=None):
        calls.append(out)
        return ReducedShard(
            shard=out,
            shard_index=0,
            shard_numel=2,
            original_shape=(4,),
            original_numel=4,
            world_size=2,
            reduce="mean",
            metadata={"output_ownership": "caller"},
        )

    result = compressed_reduce_scatter_shard(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        reduce_scatter_shard=transport,
        out=output,
    )

    assert result.shard is output
    assert calls == [output]


def test_reduce_scatter_shard_allows_async_transport_result() -> None:
    calls = []

    def transport(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.bit, op, async_op, dtype, extension_status))
        return "future-reduced-shard"

    result = compressed_reduce_scatter_shard(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        reduce_scatter_shard=transport,
        async_op=True,
        dtype="fp16",
    )

    assert result == "future-reduced-shard"
    assert calls[0][1:5] == (8, "mean", True, "fp16")


def test_reduce_scatter_shard_requires_sharded_transport() -> None:
    with pytest.raises(UnsupportedCollective, match="reduce_scatter_shard:transport"):
        compressed_reduce_scatter_shard(FakeTensor(), config=CompressionConfig())


def test_reduced_shard_exposes_logical_range_and_serializable_metadata() -> None:
    shard = ReducedShard(
        shard="rank-local-shard",
        shard_index=1,
        shard_numel=2,
        original_shape=(3,),
        original_numel=3,
        world_size=2,
        reduce="mean",
        padded_numel=4,
        dtype="fp32",
        layout="flat_contiguous",
        transport="compressed_all_to_all",
        metadata={"bucket_index": 7},
    )

    assert shard.shard_offset == 2
    assert shard.shard_end == 3
    assert shard.valid_numel == 1
    assert shard.padding_numel == 1
    assert shard.logical_range == (2, 3)
    assert shard.has_padding is True
    assert shard.is_padding_only is False
    assert shard.to_metadata() == {
        "shard_index": 1,
        "shard_numel": 2,
        "shard_offset": 2,
        "shard_end": 3,
        "valid_numel": 1,
        "original_shape": (3,),
        "original_numel": 3,
        "padded_numel": 4,
        "world_size": 2,
        "reduce": "mean",
        "dtype": "fp32",
        "layout": "flat_contiguous",
        "transport": "compressed_all_to_all",
        "metadata": {"bucket_index": 7},
    }


def test_reduced_shard_identifies_padding_only_rank() -> None:
    shard = ReducedShard(
        shard="rank-local-shard",
        shard_index=3,
        shard_numel=2,
        original_shape=(5,),
        original_numel=5,
        world_size=4,
        reduce="sum",
        padded_numel=8,
    )

    assert shard.shard_offset == 6
    assert shard.shard_end == 5
    assert shard.valid_numel == 0
    assert shard.padding_numel == 2
    assert shard.is_padding_only is True
