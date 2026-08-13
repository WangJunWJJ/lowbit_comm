from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    CompressedAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    ReduceSum,
    RuntimeBindings,
    NativeAllReduce,
    FullPrecisionWire,
)


def _program(*, compact: bool = False) -> CommunicationProgram:
    return CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=compact),
        algorithm=CompressedReduceScatterAllGather(),
    )


def _context() -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=4,
        shape=(131_073,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
    )


def _native() -> object:
    return SimpleNamespace(
        inplace_quantize=lambda *args: None,
        inplace_dequantize_reduce_mean=lambda *args: True,
        inplace_dequantize_reduce_mean_requantize=lambda *args: True,
        inplace_dequantize_gathered=lambda *args: True,
        QuantType=SimpleNamespace(Linear=object()),
        DType=SimpleNamespace(FP16=object()),
    )


def test_fulltensor_lowering_has_exactly_two_quantized_collectives() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    collective_stages = [stage for stage in lowered.stages if stage.collective]
    assert [stage.name for stage in collective_stages] == [
        "quantized_reduce_scatter",
        "quantized_all_gather",
    ]
    assert all(stage.wire == _program().wire for stage in collective_stages)
    assert [stage.name for stage in lowered.stages][-1] == "gathered_dequant_writeback"


def test_fulltensor_compile_rejects_unfused_compact_wire() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    lowered = backend.lower(_program(compact=True), _context(), RuntimeBindings())

    with pytest.raises(RuntimeError, match="compact=False"):
        backend.compile(lowered)


def test_fulltensor_compile_requires_both_fused_native_symbols() -> None:
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(
            True,
            SimpleNamespace(
                inplace_quantize=lambda *args: None,
                inplace_dequantize_reduce_mean_requantize=lambda *args: True,
                QuantType=SimpleNamespace(Linear=object()),
                DType=SimpleNamespace(FP16=object()),
            ),
        )
    )
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    with pytest.raises(RuntimeError, match="inplace_dequantize_gathered"):
        backend.compile(lowered)


def test_fulltensor_wait_does_not_schedule_collectives_or_kernels() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")
    wait_body = source.split("class _FullTensorWork:", 1)[1].split(
        "def _require_module", 1
    )[0]

    assert "all_gather_into_tensor" not in wait_body
    assert "_requantize(" not in wait_body
    assert "_writeback(" not in wait_body


def test_fulltensor_completion_avoids_device_wide_synchronize() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")

    assert ".synchronize()" not in source
    assert "event.record(" in source
    assert "event.query()" in source


def test_cuda_backend_compiles_explicit_native_all_reduce() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(False, None, "unused"))
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=FullPrecisionWire(DataType.FP16),
        algorithm=NativeAllReduce(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings(process_group="group"))
    executable = backend.compile(lowered)

    assert [stage.name for stage in lowered.stages] == ["native_all_reduce"]
    assert type(executable).__name__ == "CudaNativeAllReduceExecutable"


def test_cuda_backend_compiles_explicit_compressed_all_gather() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64),
        algorithm=CompressedAllGather(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert [stage.name for stage in lowered.stages] == [
        "quantize_full_contribution",
        "compressed_all_gather",
        "fused_dequant_reduce_mean",
    ]
    assert type(backend.compile(lowered)).__name__ == "CudaCompressedAllGatherExecutable"


@pytest.mark.parametrize(
    ("operation", "expected_divisor", "stage_suffix"),
    [(ReduceSum(), 1, "sum"), (ReduceMean(), 4, "mean")],
)
def test_fulltensor_lowering_preserves_reduction_semantics(
    operation: object,
    expected_divisor: int,
    stage_suffix: str,
) -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    program = CommunicationProgram(
        operation=operation,
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=False),
        algorithm=CompressedReduceScatterAllGather(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert lowered.reduction.divisor == expected_divisor
    assert f"reduce_{stage_suffix}" in lowered.stages[2].name
