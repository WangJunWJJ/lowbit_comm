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
        extension_status=object(),
        process_group="group",
        torch=object(),
        dist=object(),
    )

    assert executable.dtype is DataType.FP16
    assert executable.wire == QuantizedWire(8, 64)
    assert executable.layout_generation == 3


def test_dynamic_gather_never_uses_python_object_collective() -> None:
    source = Path(__file__).parents[3] / "src/lowbit_comm/backends/cuda/dynamic_all_gather.py"
    text = source.read_text(encoding="utf-8")

    assert "all_gather_object" not in text
    assert "all_gather_into_tensor" in text


def test_dynamic_payload_stride_preserves_native_kernel_alignment() -> None:
    assert aligned_payload_stride((66, 132)) == 144
    assert aligned_payload_stride((80, 64)) == 80
