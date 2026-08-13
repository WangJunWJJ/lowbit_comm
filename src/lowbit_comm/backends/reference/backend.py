"""Deterministic dependency-free backend for semantic validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lowbit_comm.core import (
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    NativeAllReduce,
    compile_reduction,
)
from lowbit_comm.core.backend import BackendCapabilities
from lowbit_comm.core.context import CompileContext, RuntimeBindings
from lowbit_comm.core.lowered import LoweredProgram, LoweredStage
from lowbit_comm.core.program import CommunicationProgram
from lowbit_comm.runtime import CompletionWork


_STAGES: dict[type[object], tuple[str, ...]] = {
    NativeAllReduce: ("native_all_reduce",),
    CompressedAllGather: ("compressed_all_gather",),
    CompressedReduceScatter: ("quantized_reduce_scatter",),
    CompressedReduceScatterAllGather: (
        "quantized_reduce_scatter",
        "quantized_all_gather",
    ),
}


@dataclass(frozen=True, slots=True)
class _ReferenceExecutable:
    lowered: LoweredProgram

    def run(self, value: Any) -> CompletionWork[Any]:
        result = list(value) if isinstance(value, list) else value
        return CompletionWork(result)


class ReferenceBackend:
    """Reference target that exposes lowering semantics without device code."""

    name = "reference"
    abi_version = 1

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        del context
        return BackendCapabilities(
            target=self.name,
            supported_bits=frozenset({4, 8}),
        )

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram:
        try:
            stage_names = _STAGES[type(program.algorithm)]
        except KeyError as error:
            raise ValueError(
                f"reference backend cannot lower {type(program.algorithm).__name__}"
            ) from error
        stages = tuple(LoweredStage(name, program.wire) for name in stage_names)
        return LoweredProgram(
            self.name,
            program,
            stages,
            compile_reduction(program.operation, context.world_size),
            context,
            bindings,
        )

    def compile(self, lowered: LoweredProgram) -> _ReferenceExecutable:
        if lowered.target != self.name:
            raise ValueError(f"cannot compile target {lowered.target!r}")
        return _ReferenceExecutable(lowered)
