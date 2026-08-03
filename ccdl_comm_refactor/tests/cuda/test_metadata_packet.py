from __future__ import annotations

import pytest


class FakePacket:
    def __init__(self, values, *, dtype="torch.int64", device="cpu") -> None:
        self.values = list(values)
        self.dtype = dtype
        self.device = device

    def numel(self) -> int:
        return len(self.values)

    def copy_(self, source):
        self.values[:] = source.values
        return self

    def tolist(self):
        return list(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value) -> None:
        self.values[index] = value


class FakeTorch:
    int64 = "torch.int64"
    tensor_calls = 0

    @classmethod
    def tensor(cls, values, *, dtype, device):
        cls.tensor_calls += 1
        return FakePacket(values, dtype=dtype, device=device)

    @staticmethod
    def empty(size, *, dtype, device="cpu"):
        return FakePacket([0] * size, dtype=dtype, device=device)


def test_metadata_packet_has_fixed_int64_layout_and_round_trips() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_NUMEL,
        decode_metadata_packet,
        encode_metadata_packet,
    )

    packet = encode_metadata_packet(
        shape=(0, 63),
        dtype="fp16",
        payload_numel=66,
        torch=FakeTorch,
        device="cpu",
    )

    assert packet.dtype == FakeTorch.int64
    assert packet.numel() == METADATA_PACKET_NUMEL
    assert decode_metadata_packet(packet) == {
        "protocol_version": 1,
        "shape": (0, 63),
        "dtype": "fp16",
        "payload_numel": 66,
        "flags": 0,
    }


@pytest.mark.parametrize("shape", [(), (0,), (63,), (64,), (65,)])
def test_metadata_packet_supports_scalar_zero_and_boundary_shapes(shape) -> None:
    from ccdl_comm.cuda.metadata_packet import (
        decode_metadata_packet,
        encode_metadata_packet,
    )

    packet = encode_metadata_packet(
        shape=shape,
        dtype="bf16",
        payload_numel=0,
        torch=FakeTorch,
        device="cpu",
    )

    assert decode_metadata_packet(packet)["shape"] == shape


def test_metadata_packet_reuses_caller_owned_output() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_NUMEL,
        encode_metadata_packet,
    )

    output = FakeTorch.empty(METADATA_PACKET_NUMEL, dtype=FakeTorch.int64)
    FakeTorch.tensor_calls = 0
    packet = encode_metadata_packet(
        shape=(65,),
        dtype="fp32",
        payload_numel=132,
        torch=FakeTorch,
        device="cpu",
        output=output,
    )

    assert packet is output
    assert FakeTorch.tensor_calls == 0


def test_metadata_packet_batch_decode_reads_gathered_tensor_once() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_NUMEL,
        decode_metadata_packets,
        encode_metadata_packet,
    )

    packets = [
        encode_metadata_packet(
            shape=(size,),
            dtype="fp16",
            payload_numel=payload,
            torch=FakeTorch,
            device="cpu",
        )
        for size, payload in ((0, 0), (65, 132))
    ]
    gathered = FakePacket(
        [value for packet in packets for value in packet.values]
    )
    calls = 0
    original_tolist = gathered.tolist

    def counting_tolist():
        nonlocal calls
        calls += 1
        return original_tolist()

    gathered.tolist = counting_tolist

    decoded = decode_metadata_packets(gathered, world_size=2)

    assert calls == 1
    assert len(gathered.values) == 2 * METADATA_PACKET_NUMEL
    assert [item["shape"] for item in decoded] == [(0,), (65,)]


def test_metadata_packet_rejects_shape_above_fixed_capacity() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_MAX_NDIM,
        encode_metadata_packet,
    )

    with pytest.raises(ValueError, match="maximum rank"):
        encode_metadata_packet(
            shape=(1,) * (METADATA_PACKET_MAX_NDIM + 1),
            dtype="fp16",
            payload_numel=1,
            torch=FakeTorch,
            device="cpu",
        )


def test_metadata_packet_rejects_unknown_dtype() -> None:
    from ccdl_comm.cuda.metadata_packet import encode_metadata_packet

    with pytest.raises(ValueError, match="unsupported metadata dtype"):
        encode_metadata_packet(
            shape=(1,),
            dtype="complex64",
            payload_numel=1,
            torch=FakeTorch,
            device="cpu",
        )


def test_metadata_packet_rejects_unknown_protocol_version() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        decode_metadata_packet,
        encode_metadata_packet,
    )

    packet = encode_metadata_packet(
        shape=(1,),
        dtype="fp16",
        payload_numel=1,
        torch=FakeTorch,
        device="cpu",
    )
    packet[0] = 2

    with pytest.raises(RuntimeError, match="protocol version"):
        decode_metadata_packet(packet)


def test_metadata_packet_rejects_nonzero_reserved_fields() -> None:
    from ccdl_comm.cuda.metadata_packet import (
        METADATA_PACKET_RESERVED_OFFSET,
        decode_metadata_packet,
        encode_metadata_packet,
    )

    packet = encode_metadata_packet(
        shape=(1,),
        dtype="fp16",
        payload_numel=1,
        torch=FakeTorch,
        device="cpu",
    )
    packet[METADATA_PACKET_RESERVED_OFFSET] = 1

    with pytest.raises(RuntimeError, match="reserved"):
        decode_metadata_packet(packet)
