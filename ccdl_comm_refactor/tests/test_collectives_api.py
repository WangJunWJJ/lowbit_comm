from ccdl_comm.collectives import ImmediateWork, UnsupportedCollective, compressed_all_reduce
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16"):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def __truediv__(self, value):
        return FakeTensor([item / value for item in self.values], dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values and self.dtype == other.dtype


def test_compressed_all_reduce_rejects_unsupported_strategy() -> None:
    config = CompressionConfig()

    try:
        compressed_all_reduce(FakeTensor([1.0]), config=config, strategy="ring")
    except UnsupportedCollective as exc:
        assert "all_reduce:ring" in str(exc)
    else:
        raise AssertionError("expected UnsupportedCollective")


def test_compressed_all_reduce_can_return_immediate_work() -> None:
    config = CompressionConfig()
    calls = []

    def quantize(tensor, active_config):
        calls.append(("quantize", tensor, active_config))
        return {"buffer": tensor}

    def all_reduce(payload, op):
        calls.append(("all_reduce", payload.buffer, op))
        return payload

    def dequantize(payload, shape, active_config, dtype):
        calls.append(("dequantize", payload.buffer, shape, active_config, dtype))
        return payload.buffer

    work = compressed_all_reduce(
        FakeTensor([2.0]),
        config=config,
        op="mean",
        strategy="all_reduce",
        async_op=True,
        world_size=1,
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=all_reduce,
    )

    assert isinstance(work, ImmediateWork)
    assert work.wait() == FakeTensor([2.0])
    assert calls == [
        ("quantize", FakeTensor([2.0]), config),
        ("all_reduce", FakeTensor([2.0]), "sum"),
        ("dequantize", FakeTensor([2.0]), (1,), config, "fp16"),
    ]


def test_compressed_all_reduce_blocking_mean_divides_by_world_size_for_all_reduce_strategy() -> None:
    config = CompressionConfig()

    result = compressed_all_reduce(
        FakeTensor([4.0]),
        config=config,
        op="mean",
        strategy="all_reduce",
        world_size=2,
        quantize=lambda tensor, active_config: {"buffer": tensor},
        all_reduce=lambda payload, op: payload,
        dequantize=lambda payload, shape, active_config, dtype: payload.buffer,
    )

    assert result == FakeTensor([2.0])
