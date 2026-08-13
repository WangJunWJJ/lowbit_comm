from __future__ import annotations

import os

import torch

from lowbit_comm.backends.cuda.codec import decode_dynamic_metadata_into
from lowbit_comm.backends.cuda.loader import load_cuda_extension
from lowbit_comm.core import DataType, MetadataPacket, QuantizedWire


def main() -> None:
    status = load_cuda_extension(
        module_name=os.environ.get(
            "LOWBIT_COMM_CUDA_EXTENSION_MODULE",
            "lowbit_comm_cuda_ops",
        )
    )
    if not status.available:
        raise RuntimeError(status.reason)
    wire = QuantizedWire(8, 64)
    packet = MetadataPacket((65,), DataType.FP16, wire, 132, 7)
    metadata = torch.tensor(packet.to_values(), dtype=torch.int64, device="cuda")
    descriptors = torch.empty((1, 12), dtype=torch.int64, device="cuda")
    decode_dynamic_metadata_into(
        metadata,
        descriptors,
        world_size=1,
        dtype=DataType.FP16,
        wire=wire,
        layout_generation=7,
        max_numel=4096,
        payload_stride=144,
        extension_status=status,
    )
    assert int(descriptors[0, 0]) == 0
    metadata[0] = 99
    decode_dynamic_metadata_into(
        metadata,
        descriptors,
        world_size=1,
        dtype=DataType.FP16,
        wire=wire,
        layout_generation=7,
        max_numel=4096,
        payload_stride=144,
        extension_status=status,
    )
    assert int(descriptors[0, 0]) != 0
    print("METADATA_KERNEL_VALIDATION_OK")


if __name__ == "__main__":
    main()
