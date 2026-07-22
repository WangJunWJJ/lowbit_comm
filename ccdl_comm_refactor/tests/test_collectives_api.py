from ccdl_comm.collectives import (
    GatheredPayloads,
    ImmediateWork,
    UnsupportedCollective,
    compressed_all_gather,
    compressed_all_reduce,
)
from ccdl_comm.collectives.all_reduce import _make_payload_all_gather
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16"):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def __truediv__(self, value):
        return FakeTensor([item / value for item in self.values], dtype=self.dtype)

    def __add__(self, other):
        return FakeTensor([left + right for left, right in zip(self.values, other.values)], dtype=self.dtype)

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


def test_compressed_all_reduce_defaults_to_all_gather_reduce_strategy() -> None:
    config = CompressionConfig()
    calls = []

    def quantize(tensor, active_config):
        calls.append(("quantize", tensor, active_config))
        return tensor

    def all_gather(payload):
        calls.append(("all_gather", payload))
        return GatheredPayloads(payloads=[FakeTensor([2.0]), FakeTensor([4.0])], world_size=2)

    def dequantize(payload, shape, active_config, dtype):
        calls.append(("dequantize", payload, shape, active_config, dtype))
        return payload

    result = compressed_all_reduce(
        FakeTensor([1.0]),
        config=config,
        quantize=quantize,
        dequantize=dequantize,
        all_gather=all_gather,
    )

    assert result == FakeTensor([3.0])
    assert calls == [
        ("quantize", FakeTensor([1.0]), config),
        ("all_gather", FakeTensor([1.0])),
        ("dequantize", FakeTensor([2.0]), (1,), config, "fp16"),
        ("dequantize", FakeTensor([4.0]), (1,), config, "fp16"),
    ]


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


def test_payload_all_gather_transport_gathers_payload_buffers_and_restores_metadata() -> None:
    calls = []

    def buffer_all_gather(buffer):
        calls.append(buffer)
        return GatheredPayloads(payloads=[FakeTensor([1.0]), FakeTensor([2.0])], world_size=2)

    payload_all_gather = _make_payload_all_gather(buffer_all_gather)
    result = payload_all_gather(CompressedPayload(buffer=FakeTensor([0.0]), shape=(1,), dtype="fp16"))

    assert calls == [FakeTensor([0.0])]
    assert result.world_size == 2
    assert result.payloads == [
        CompressedPayload(buffer=FakeTensor([1.0]), shape=(1,), dtype="fp16"),
        CompressedPayload(buffer=FakeTensor([2.0]), shape=(1,), dtype="fp16"),
    ]


def test_compressed_all_gather_returns_decompressed_rank_tensors() -> None:
    config = CompressionConfig()
    calls = []

    def quantize(tensor, active_config):
        calls.append(("quantize", tensor, active_config))
        return CompressedPayload(buffer=tensor, shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        calls.append(("all_gather", payload))
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer=FakeTensor([1.0]), shape=(1,), dtype="fp16"),
                CompressedPayload(buffer=FakeTensor([2.0]), shape=(1,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize(payload, shape, active_config, dtype):
        calls.append(("dequantize", payload.buffer, shape, active_config, dtype))
        return payload.buffer

    result = compressed_all_gather(
        FakeTensor([0.0]),
        config=config,
        quantize=quantize,
        all_gather=all_gather,
        dequantize=dequantize,
    )

    assert result == [FakeTensor([1.0]), FakeTensor([2.0])]
    assert calls == [
        ("quantize", FakeTensor([0.0]), config),
        ("all_gather", CompressedPayload(buffer=FakeTensor([0.0]), shape=(1,), dtype="fp16")),
        ("dequantize", FakeTensor([1.0]), (1,), config, "fp16"),
        ("dequantize", FakeTensor([2.0]), (1,), config, "fp16"),
    ]


def test_compressed_all_gather_can_return_immediate_work() -> None:
    config = CompressionConfig()

    work = compressed_all_gather(
        FakeTensor([3.0]),
        config=config,
        async_op=True,
        quantize=lambda tensor, active_config: CompressedPayload(buffer=tensor, shape=tensor.shape, dtype="fp16"),
        all_gather=lambda payload: GatheredPayloads(payloads=[payload], world_size=1),
        dequantize=lambda payload, shape, active_config, dtype: payload.buffer,
    )

    assert isinstance(work, ImmediateWork)
    assert work.wait() == [FakeTensor([3.0])]
