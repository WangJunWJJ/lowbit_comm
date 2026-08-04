import pytest

from ccdl_comm import CompressionConfig, compressed_hierarchical_all_reduce
from ccdl_comm.exceptions import UnsupportedCollective


class FakeTensor:
    shape = (4,)
    dtype = "float32"


def test_hierarchical_all_reduce_uses_injected_transport() -> None:
    calls = []

    def hierarchical_transport(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.bit, op, async_op, dtype, extension_status))
        return "hierarchical-result"

    result = compressed_hierarchical_all_reduce(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        op="mean",
        hierarchical_all_reduce=hierarchical_transport,
    )

    assert result == "hierarchical-result"
    assert calls[0][1:5] == (8, "mean", False, "auto")


def test_hierarchical_all_reduce_falls_back_without_transport() -> None:
    calls = []

    def fallback(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.group_size, op, async_op, dtype, extension_status))
        return "fallback-result"

    result = compressed_hierarchical_all_reduce(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=32),
        all_gather_fallback=fallback,
    )

    assert result == "fallback-result"
    assert calls[0][1:5] == (32, "mean", False, "auto")


def test_hierarchical_all_reduce_rejects_unsupported_op() -> None:
    with pytest.raises(UnsupportedCollective, match="hierarchical:max"):
        compressed_hierarchical_all_reduce(FakeTensor(), config=CompressionConfig(), op="max")


def test_hierarchical_all_reduce_requires_transport_or_fallback() -> None:
    with pytest.raises(UnsupportedCollective, match="hierarchical:transport"):
        compressed_hierarchical_all_reduce(FakeTensor(), config=CompressionConfig())
