from __future__ import annotations

from pathlib import Path

from lowbit_comm.backends.cuda.dynamic_all_gather import (
    CudaDynamicAllGather,
    aligned_payload_stride,
)
from lowbit_comm.core import DataType, QuantizedWire


def test_dynamic_gather_binds_dtype_wire_and_layout_generation() -> None:
    executable = CudaDynamicAllGather(
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        layout_generation=3,
        max_numel=4096,
        extension_status=object(),
        process_group="group",
        torch=object(),
        dist=object(),
    )

    assert executable.dtype is DataType.FP16
    assert executable.wire == QuantizedWire(8, 64)
    assert executable.layout_generation == 3
    assert executable.max_numel == 4096


def test_dynamic_gather_never_uses_python_object_collective() -> None:
    source = Path(__file__).parents[3] / "src/lowbit_comm/backends/cuda/dynamic_all_gather.py"
    text = source.read_text(encoding="utf-8")

    assert "all_gather_object" not in text
    assert "all_gather_into_tensor" in text
    assert ".tolist()" not in text
    assert "pin_memory=True" in text
    assert "non_blocking=True" in text
    assert "decode_dynamic_metadata_into(" in text
    assert "MetadataPacket.from_values" not in text


def test_bounded_dynamic_path_queues_payload_before_host_decode() -> None:
    source = Path(__file__).parents[3] / "src/lowbit_comm/backends/cuda/dynamic_all_gather.py"
    text = source.read_text(encoding="utf-8")

    payload_collective = text.index("gathered_payload,")
    host_decode = text.index("_decode_descriptors(")
    assert payload_collective < host_decode


def test_dynamic_gather_rejects_invalid_shape_bound() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_numel"):
        CudaDynamicAllGather(
            dtype=DataType.FP16,
            wire=QuantizedWire(8, 64),
            layout_generation=3,
            max_numel=0,
            extension_status=object(),
            torch=object(),
            dist=object(),
        )


def test_dynamic_payload_stride_preserves_native_kernel_alignment() -> None:
    assert aligned_payload_stride((66, 132)) == 144
    assert aligned_payload_stride((80, 64)) == 80
