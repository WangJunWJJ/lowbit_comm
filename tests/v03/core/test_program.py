from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lowbit_comm.core import (
    AutoAlgorithm,
    CommunicationProgram,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    ErrorFeedbackDomain,
    FullPrecisionWire,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
)


def test_program_keeps_output_and_wire_orthogonal() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(dtype=DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    assert program.output.dtype is DataType.FP16
    assert program.wire.bit == 8
    assert not hasattr(program, "restore_mode")


def test_program_is_immutable() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=ReducedShard(dtype=DataType.BF16, layout_version=2),
        wire=QuantizedWire(bit=8, group_size=32),
        algorithm=CompressedReduceScatter(),
        error_feedback=ErrorFeedbackDomain.GRADIENT,
    )

    with pytest.raises(FrozenInstanceError):
        program.async_op = False


@pytest.mark.parametrize("bit", [0, 3, 16])
def test_quantized_wire_rejects_unsupported_bits(bit: int) -> None:
    with pytest.raises(ValueError, match="bit"):
        QuantizedWire(bit=bit, group_size=64)


@pytest.mark.parametrize("group_size", [0, 8, 128])
def test_quantized_wire_rejects_unsupported_group_sizes(group_size: int) -> None:
    with pytest.raises(ValueError, match="group_size"):
        QuantizedWire(bit=8, group_size=group_size)


def test_full_precision_wire_requires_typed_dtype() -> None:
    with pytest.raises(TypeError, match="dtype"):
        FullPrecisionWire(dtype="fp16")


def test_auto_is_an_explicit_algorithm_value() -> None:
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP32),
        wire=FullPrecisionWire(DataType.FP32),
        algorithm=AutoAlgorithm(),
    )

    assert isinstance(program.algorithm, AutoAlgorithm)
