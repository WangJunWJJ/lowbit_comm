import pytest

from ccdl_comm import CompressionConfig, compressed_reduce_scatter
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
