from __future__ import annotations

import pytest

from lowbit_comm.core import DataType, MetadataPacket, QuantizedWire
from lowbit_comm.core.metadata import METADATA_PACKET_WORDS


def test_metadata_packet_round_trips_every_wire_field() -> None:
    packet = MetadataPacket(
        shape=(3, 5, 7),
        dtype=DataType.BF16,
        wire=QuantizedWire(4, 32, quant_type="e2m1", compact=True),
        payload_numel=96,
        layout_generation=9,
        flags=5,
    )

    values = packet.to_values()

    assert len(values) == METADATA_PACKET_WORDS
    assert MetadataPacket.from_values(values) == packet


def test_metadata_packet_size_is_fixed_across_runtime_shapes() -> None:
    scalar = MetadataPacket(
        shape=(),
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        payload_numel=66,
        layout_generation=0,
    )
    matrix = MetadataPacket(
        shape=(2, 4),
        dtype=DataType.FP32,
        wire=QuantizedWire(8, 16, compact=False),
        payload_numel=144,
        layout_generation=1,
    )

    assert len(scalar.to_values()) == len(matrix.to_values()) == 24


def test_metadata_packet_rejects_unknown_version_and_nonzero_reserved_words() -> None:
    packet = MetadataPacket(
        shape=(64,),
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        payload_numel=66,
        layout_generation=0,
    )
    bad_version = list(packet.to_values())
    bad_version[0] += 1
    bad_reserved = list(packet.to_values())
    bad_reserved[12] = 1

    with pytest.raises(ValueError, match="protocol version"):
        MetadataPacket.from_values(bad_version)
    with pytest.raises(ValueError, match="reserved"):
        MetadataPacket.from_values(bad_reserved)


def test_metadata_packet_rejects_shapes_above_fixed_capacity() -> None:
    with pytest.raises(ValueError, match="maximum rank"):
        MetadataPacket(
            shape=(1,) * 9,
            dtype=DataType.FP16,
            wire=QuantizedWire(8, 64),
            payload_numel=66,
            layout_generation=0,
        )
