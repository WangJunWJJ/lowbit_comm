import pytest

from ccdl_comm.communication.transport_capability import (
    CompressedTransportCapability,
    bind_compressed_transport,
    capability_for,
    require_compressed_transport,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective


def _capability() -> CompressedTransportCapability:
    return CompressedTransportCapability(
        codec="ccdl",
        collectives=frozenset({"all_reduce"}),
        bits=frozenset({8}),
        group_sizes=frozenset({64}),
        dtypes=frozenset({"fp16"}),
        output_layouts=frozenset({"full"}),
    )


def test_bind_compressed_transport_preserves_capability_and_call() -> None:
    transport = bind_compressed_transport(lambda value, op: (value, op), _capability())

    assert capability_for(transport) == _capability()
    assert transport("payload", "sum") == ("payload", "sum")


def test_require_compressed_transport_rejects_missing_declaration() -> None:
    with pytest.raises(UnsupportedCollective, match="compressed payload capability"):
        require_compressed_transport(
            lambda payload, op: payload,
            collective="all_reduce",
            config=CompressionConfig(),
            dtype="fp16",
            output_layout="full",
        )


def test_require_compressed_transport_rejects_profile_mismatch() -> None:
    transport = bind_compressed_transport(lambda payload, op: payload, _capability())

    with pytest.raises(UnsupportedCollective, match="dtype=fp32"):
        require_compressed_transport(
            transport,
            collective="all_reduce",
            config=CompressionConfig(),
            dtype="fp32",
            output_layout="full",
        )
