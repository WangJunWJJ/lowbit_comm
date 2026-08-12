"""Compile-once pipeline from Semantic IR to a bound executable."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from lowbit_comm.core.backend import BackendCapabilities, CompiledExecutable
from lowbit_comm.core.context import CompileContext, RuntimeBindings
from lowbit_comm.core.errors import UnsupportedProgram
from lowbit_comm.core.execution_info import ExecutionInfo
from lowbit_comm.core.lowered import LoweredProgram
from lowbit_comm.core.program import CommunicationProgram
from lowbit_comm.core.types import (
    AutoAlgorithm,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    FullPrecisionWire,
    NativeAllReduce,
    QuantizedWire,
)

from .cost_model import BenchmarkEvidence, decide_auto
from .registry import BackendRegistry
from .verifier import verify


_ALGORITHM_NAMES: dict[type[object], str] = {
    AutoAlgorithm: "auto",
    NativeAllReduce: "native",
    CompressedAllGather: "compressed_all_gather",
    CompressedReduceScatter: "compressed_reduce_scatter",
    CompressedReduceScatterAllGather: "compressed_rs_ag",
}


@dataclass(frozen=True, slots=True)
class BoundExecutable:
    """Hot-path object containing no registry or strategy decision logic."""

    executable: CompiledExecutable
    lowered: LoweredProgram
    execution_info: ExecutionInfo

    def run(self, value: Any) -> Any:
        return self.executable.run(value)


def compile(
    program: CommunicationProgram,
    context: CompileContext,
    *,
    bindings: RuntimeBindings,
    registry: BackendRegistry,
    evidence: BenchmarkEvidence | None = None,
) -> BoundExecutable:
    verify(program, context)
    target = context.device_type
    backend = registry.resolve(target)
    capabilities = backend.capabilities(context)
    effective, fallback_reason, evidence_id = _select_effective_program(
        program,
        context,
        target,
        evidence,
    )
    _require_supported(effective, capabilities)
    lowered = backend.lower(effective, context, bindings)
    executable = backend.compile(lowered)
    info = ExecutionInfo(
        requested_algorithm=_algorithm_name(program.algorithm),
        effective_algorithm=_algorithm_name(effective.algorithm),
        requested_wire=program.wire,
        effective_wire=effective.wire,
        fallback_reason=fallback_reason,
        evidence_id=evidence_id,
    )
    return BoundExecutable(executable, lowered, info)


def _select_effective_program(
    program: CommunicationProgram,
    context: CompileContext,
    target: str,
    evidence: BenchmarkEvidence | None,
) -> tuple[CommunicationProgram, str | None, str | None]:
    if not isinstance(program.algorithm, AutoAlgorithm):
        return program, None, None

    decision = decide_auto(target, context, evidence)
    if decision.use_compression:
        return (
            replace(program, algorithm=CompressedReduceScatterAllGather()),
            None,
            decision.evidence_id,
        )
    return (
        replace(
            program,
            algorithm=NativeAllReduce(),
            wire=FullPrecisionWire(context.dtype),
        ),
        decision.fallback_reason,
        decision.evidence_id,
    )


def _require_supported(
    program: CommunicationProgram,
    capabilities: BackendCapabilities,
) -> None:
    algorithm = _algorithm_name(program.algorithm)
    if algorithm not in capabilities.supported_algorithms:
        raise UnsupportedProgram(
            f"backend {capabilities.target!r} does not support algorithm {algorithm!r}"
        )
    if isinstance(program.wire, QuantizedWire) and (
        program.wire.bit not in capabilities.supported_bits
    ):
        raise UnsupportedProgram(
            f"backend {capabilities.target!r} does not support {program.wire.bit}-bit wire"
        )


def _algorithm_name(algorithm: object) -> str:
    try:
        return _ALGORITHM_NAMES[type(algorithm)]
    except KeyError as error:
        raise UnsupportedProgram(
            f"unknown algorithm type {type(algorithm).__name__}"
        ) from error
