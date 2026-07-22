from ccdl_comm.communication.collectives import CompressedAllReduce, CompressedPayload
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_compressed_all_reduce_orchestrates_codec_and_transport() -> None:
    calls = []
    config = CompressionConfig(bit=8)

    def compress(tensor, active_config):
        calls.append(("compress", tensor, active_config.bit))
        return CompressedPayload(buffer=("q", tensor.values), shape=tensor.shape, dtype="fp16")

    def all_reduce(payload, op):
        calls.append(("all_reduce", payload.buffer, op))
        return CompressedPayload(buffer=("reduced", payload.buffer), shape=payload.shape, dtype=payload.dtype)

    def decompress(payload, active_config):
        calls.append(("decompress", payload.buffer, payload.shape, payload.dtype, active_config.bit))
        return FakeTensor([3.0, 5.0])

    collective = CompressedAllReduce(config=config, compress=compress, all_reduce=all_reduce, decompress=decompress)

    result = collective.run(FakeTensor([1.0, 2.0]), op="sum")

    assert result == FakeTensor([3.0, 5.0])
    assert calls == [
        ("compress", FakeTensor([1.0, 2.0]), 8),
        ("all_reduce", ("q", (1.0, 2.0)), "sum"),
        ("decompress", ("reduced", ("q", (1.0, 2.0))), (2,), "fp16", 8),
    ]


def test_compressed_payload_replaces_buffer_without_losing_metadata() -> None:
    payload = CompressedPayload(buffer="before", shape=(2, 3), dtype="bf16", metadata={"scale": "grouped"})

    replaced = payload.with_buffer("after")

    assert replaced == CompressedPayload(buffer="after", shape=(2, 3), dtype="bf16", metadata={"scale": "grouped"})
