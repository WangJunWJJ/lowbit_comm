"""Static verifier for Semantic IR programs."""

from __future__ import annotations

from lowbit_comm.core.context import CompileContext
from lowbit_comm.core.errors import ProgramVerificationError
from lowbit_comm.core.program import CommunicationProgram
from lowbit_comm.core.types import (
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    ErrorFeedbackDomain,
    FullTensor,
    ReducedShard,
)


def verify(program: CommunicationProgram, context: CompileContext) -> None:
    if not isinstance(program, CommunicationProgram):
        raise TypeError("program must be a CommunicationProgram")
    if not isinstance(context, CompileContext):
        raise TypeError("context must be a CompileContext")
    if isinstance(program.output, ReducedShard) and isinstance(
        program.algorithm,
        CompressedReduceScatterAllGather,
    ):
        raise ProgramVerificationError(
            "ReducedShard output cannot use an algorithm that restores FullTensor"
        )
    if isinstance(program.output, FullTensor) and isinstance(
        program.algorithm,
        CompressedReduceScatter,
    ):
        raise ProgramVerificationError(
            "FullTensor output cannot use a shard-only reduce-scatter algorithm"
        )
    output_dtype = getattr(program.output, "dtype", None)
    if output_dtype is not context.dtype:
        raise ProgramVerificationError(
            "output dtype must match the CompileContext input dtype"
        )
    if (
        program.error_feedback is ErrorFeedbackDomain.PARAMETER_DELTA
        and isinstance(program.output, FullTensor)
    ):
        raise ProgramVerificationError(
            "parameter-delta error feedback belongs to the sharded training adapter"
        )
