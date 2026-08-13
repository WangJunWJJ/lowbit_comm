from __future__ import annotations

import pytest

from lowbit_comm.compiler import ProgramVerificationError, verify
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    ErrorFeedbackDomain,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
    RuntimeBindings,
)


CONTEXT = CompileContext(
    rank=0,
    world_size=4,
    shape=(1024,),
    dtype=DataType.FP16,
    device_type="cuda",
    device_architecture="sm86",
    topology_signature="single_node_pcie",
)


def test_compile_context_is_hashable_and_excludes_runtime_bindings() -> None:
    bindings = RuntimeBindings(process_group=object(), backend_runtime=object())

    assert hash(CONTEXT)
    assert bindings.process_group is not None
    assert not hasattr(CONTEXT, "process_group")


def test_reduced_shard_rejects_full_restore_algorithm() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=ReducedShard(DataType.FP16, layout_version=0),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    with pytest.raises(ProgramVerificationError, match="ReducedShard"):
        verify(program, CONTEXT)


def test_fulltensor_rejects_shard_only_algorithm() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatter(),
    )

    with pytest.raises(ProgramVerificationError, match="FullTensor"):
        verify(program, CONTEXT)


def test_parameter_feedback_rejects_fulltensor_gradient_program() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
        error_feedback=ErrorFeedbackDomain.PARAMETER_DELTA,
    )

    with pytest.raises(ProgramVerificationError, match="parameter-delta error feedback"):
        verify(program, CONTEXT)


def test_output_dtype_must_match_context_dtype() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.BF16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    with pytest.raises(ProgramVerificationError, match="dtype"):
        verify(program, CONTEXT)


def test_valid_quantized_fulltensor_program_passes() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
        error_feedback=ErrorFeedbackDomain.GRADIENT,
    )

    assert verify(program, CONTEXT) is None
