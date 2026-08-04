from ccdl_comm.communication.gather_reduce import CompressedAllGatherReduce, GatheredPayloads
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)

    def __add__(self, other):
        return FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __truediv__(self, value):
        return FakeTensor(a / value for a in self.values)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_compressed_all_gather_reduce_decompresses_each_payload_then_sums() -> None:
    calls = []
    config = CompressionConfig(bit=8)

    def compress(tensor, active_config):
        calls.append(("compress", tensor, active_config.bit))
        return {"rank": 0, "values": tensor.values}

    def all_gather(payload):
        calls.append(("all_gather", payload))
        return GatheredPayloads(payloads=[payload, {"rank": 1, "values": (3.0, 4.0)}], world_size=2)

    def decompress(payload, shape, active_config, dtype):
        calls.append(("decompress", payload, shape, dtype, active_config.bit))
        return FakeTensor(payload["values"])

    collective = CompressedAllGatherReduce(
        config=config,
        compress=compress,
        all_gather=all_gather,
        decompress=decompress,
    )

    result = collective.run(FakeTensor([1.0, 2.0]), shape=(2,), dtype="fp16", reduce="sum")

    assert result == FakeTensor([4.0, 6.0])
    assert calls == [
        ("compress", FakeTensor([1.0, 2.0]), 8),
        ("all_gather", {"rank": 0, "values": (1.0, 2.0)}),
        ("decompress", {"rank": 0, "values": (1.0, 2.0)}, (2,), "fp16", 8),
        ("decompress", {"rank": 1, "values": (3.0, 4.0)}, (2,), "fp16", 8),
    ]


def test_compressed_all_gather_reduce_can_average_like_ddp() -> None:
    collective = CompressedAllGatherReduce(
        config=CompressionConfig(bit=8),
        compress=lambda tensor, config: tensor,
        all_gather=lambda payload: GatheredPayloads(payloads=[FakeTensor([2.0]), FakeTensor([4.0])], world_size=2),
        decompress=lambda payload, shape, config, dtype: payload,
    )

    assert collective.run(FakeTensor([0.0]), shape=(1,), dtype="fp16", reduce="mean") == FakeTensor([3.0])
