import pytest

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.gather_reduce import GatheredPayloads
from ccdl_comm.communication.payload_packing import make_fused_payload_all_gather


def test_fused_payload_all_gather_packs_buffer_and_tensor_metadata_once() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    local_payload = CompressedPayload(
        buffer=torch.tensor([-2, 3, 7, 11], dtype=torch.int8),
        shape=(4,),
        dtype="fp16",
        metadata={
            "scales": torch.tensor([0.25, 0.5], dtype=torch.float16),
            "original_numel": 4,
        },
    )
    remote_payload = CompressedPayload(
        buffer=torch.tensor([1, 2, 3, 4], dtype=torch.int8),
        shape=(4,),
        dtype="fp16",
        metadata={
            "scales": torch.tensor([1.0, 2.0], dtype=torch.float16),
            "original_numel": 4,
        },
    )

    def byte_all_gather(packed):
        calls.append((packed.dtype, tuple(packed.shape)))
        packer = byte_all_gather.packer
        return GatheredPayloads(
            payloads=[
                packed,
                packer.pack(remote_payload),
            ],
            world_size=2,
        )

    fused_all_gather = make_fused_payload_all_gather(byte_all_gather)
    byte_all_gather.packer = fused_all_gather.packer

    gathered = fused_all_gather(local_payload)

    assert calls == [(torch.uint8, tuple(fused_all_gather.packer.pack(local_payload).shape))]
    assert gathered.world_size == 2
    assert [payload.buffer.tolist() for payload in gathered.payloads] == [
        [-2, 3, 7, 11],
        [1, 2, 3, 4],
    ]
    assert [payload.metadata["scales"].tolist() for payload in gathered.payloads] == [
        [0.25, 0.5],
        [1.0, 2.0],
    ]
    assert [payload.metadata["original_numel"] for payload in gathered.payloads] == [4, 4]


def test_fused_payload_all_gather_rejects_missing_tensor_metadata() -> None:
    torch = pytest.importorskip("torch")
    fused_all_gather = make_fused_payload_all_gather(
        lambda packed: GatheredPayloads(payloads=[packed], world_size=1)
    )
    payload = CompressedPayload(
        buffer=torch.tensor([1, 2, 3], dtype=torch.int8),
        shape=(3,),
        dtype="fp16",
        metadata={"original_numel": 3},
    )

    with pytest.raises(ValueError, match="requires tensor metadata"):
        fused_all_gather(payload)
