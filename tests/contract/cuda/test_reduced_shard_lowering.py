from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatter,
    DataType,
    QuantizedWire,
    ReduceMean,
    ReduceSum,
    ReducedShard,
    ReducedShardValue,
    RuntimeBindings,
    WorkspaceRole,
)


def _program() -> CommunicationProgram:
    return CommunicationProgram(
        operation=ReduceMean(),
        output=ReducedShard(DataType.FP16, layout_version=3),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatter(),
    )


def _context(*, numel: int = 130, rank: int = 1) -> CompileContext:
    return CompileContext(
        rank=rank,
        world_size=4,
        shape=(numel,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
    )


def _native_module() -> object:
    return SimpleNamespace(
        inplace_quantize=lambda *args: None,
        inplace_dequantize_reduce_mean=lambda *args: True,
        QuantType=SimpleNamespace(Linear=object()),
    )


def test_reduced_shard_lowering_has_no_full_gather_stage() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native_module()))
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    names = [stage.name for stage in lowered.stages]
    assert names == [
        "quantize_destination_chunks",
        "quantized_reduce_scatter",
        "fused_dequant_reduce_mean",
        "return_reduced_shard",
    ]
    assert not any("all_gather" in name for name in names)
    assert lowered.context == _context()


def test_static_shard_plan_aligns_each_destination_chunk_to_group_size() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native_module()))
    lowered = backend.lower(_program(), _context(), RuntimeBindings())
    executable = backend.compile(lowered)

    assert executable.plan.original_numel == 130
    assert executable.plan.shard_numel == 64
    assert executable.plan.padded_numel == 256
    assert executable.plan.logical_range == (64, 128)
    assert executable.payload_numel == 66
    assert executable.payload_stride == 80
    assert executable.payload_stride % 16 == 0


def test_reduced_shard_value_exposes_valid_range_without_gathering() -> None:
    value = ReducedShardValue(
        tensor=object(),
        shard_index=3,
        shard_numel=64,
        original_shape=(130,),
        original_numel=130,
        world_size=4,
        reduction="mean",
        dtype=DataType.FP16,
        layout_version=3,
    )

    assert value.logical_range == (130, 130)
    assert value.valid_numel == 0
    assert value.padding_numel == 64


def test_cuda_compile_rejects_missing_fused_kernel_at_compile_time() -> None:
    module = SimpleNamespace(inplace_quantize=lambda *args: None)
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, module))
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    with pytest.raises(RuntimeError, match="inplace_dequantize_reduce_mean"):
        backend.compile(lowered)


def test_reduced_shard_sum_lowering_uses_no_normalization() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native_module()))
    program = CommunicationProgram(
        operation=ReduceSum(),
        output=ReducedShard(DataType.FP16, layout_version=3),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatter(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert lowered.reduction.name == "sum"
    assert lowered.reduction.divisor == 1
    assert lowered.stages[2].name == "fused_dequant_reduce_sum"


def test_reduced_shard_completion_records_output_ready_event() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")
    reduced_body = source.split("class CudaReducedShardExecutable:", 1)[1].split(
        "class CudaFullTensorExecutable:", 1
    )[0]

    assert "CompletionOutcome" in reduced_body
    assert "event.record(" in reduced_body


def test_reduced_shard_lowering_contains_compiled_internal_buffer_plan() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native_module()))

    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    roles = {buffer.role for buffer in lowered.buffer_plan.buffers}
    assert roles == {
        WorkspaceRole.PADDED_INPUT,
        WorkspaceRole.SEND,
        WorkspaceRole.RECEIVE,
    }
    assert lowered.buffer_plan.total_bytes > 0
    assert WorkspaceRole.OUTPUT not in roles
