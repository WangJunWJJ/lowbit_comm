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
from lowbit_comm.core.backend import BackendCapabilities, CapabilitySpec
from lowbit_comm.core.context import CompileContext, RuntimeBindings
from lowbit_comm.core.lowered import ExecutorKind, LoweredProgram, LoweredStage
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

_EXECUTORS: dict[type[object], ExecutorKind] = {
    NativeAllReduce: ExecutorKind.NATIVE_ALL_REDUCE,
    CompressedAllGather: ExecutorKind.COMPRESSED_ALL_GATHER,
    CompressedReduceScatter: ExecutorKind.REDUCED_SHARD,
    CompressedReduceScatterAllGather: ExecutorKind.COMPRESSED_RS_AG,
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
        return BackendCapabilities(
            target=self.name,
            specifications=_reference_capabilities(context),
            backend_abi_version=self.abi_version,
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
            _EXECUTORS[type(program.algorithm)],
        )

    def compile(self, lowered: LoweredProgram) -> _ReferenceExecutable:
        if lowered.target != self.name:
            raise ValueError(f"cannot compile target {lowered.target!r}")
        return _ReferenceExecutable(lowered)


def _reference_capabilities(context: CompileContext) -> tuple[CapabilitySpec, ...]:
    specifications: list[CapabilitySpec] = []
    for operation in ("sum", "mean"):
        specifications.append(
            CapabilitySpec(
                operation=operation,
                output="full_tensor",
                wire="full_precision",
                algorithm="native",
                dtype=context.dtype,
                physical_primitive="reference_all_reduce",
            )
        )
        for bit in (4, 8):
            for group_size in (16, 32, 64):
                for compact in (False, True):
                    common = dict(
                        operation=operation,
                        wire="quantized",
                        dtype=context.dtype,
                        bit=bit,
                        group_size=group_size,
                        quant_type="linear",
                        compact=compact,
                    )
                    specifications.extend(
                        (
                            CapabilitySpec(
                                **common,
                                output="full_tensor",
                                algorithm="compressed_all_gather",
                                physical_primitive="reference_all_gather",
                            ),
                            CapabilitySpec(
                                **common,
                                output="reduced_shard",
                                algorithm="compressed_reduce_scatter",
                                physical_primitive="reference_reduce_scatter",
                            ),
                            CapabilitySpec(
                                **common,
                                output="full_tensor",
                                algorithm="compressed_rs_ag",
                                physical_primitive="reference_rs_ag",
                            ),
                        )
                    )
    return tuple(specifications)
