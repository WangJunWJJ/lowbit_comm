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
    FullTensor,
    HierarchicalCompressed,
    ReducedShard,
)
from lowbit_comm.core.operations import ReduceMean, ReduceSum

from .cost_model import BenchmarkEvidence, decide_auto
from .evidence import EvidenceCatalog
from .registry import BackendRegistry
from .verifier import verify


_ALGORITHM_NAMES: dict[type[object], str] = {
    AutoAlgorithm: "auto",
    NativeAllReduce: "native",
    CompressedAllGather: "compressed_all_gather",
    CompressedReduceScatter: "compressed_reduce_scatter",
    CompressedReduceScatterAllGather: "compressed_rs_ag",
    HierarchicalCompressed: "hierarchical_compressed",
}


@dataclass(frozen=True, slots=True)
class BoundExecutable:
    """Hot-path object containing no registry or strategy decision logic."""

    executable: CompiledExecutable
    lowered: LoweredProgram
    execution_info: ExecutionInfo

    def run(self, value: Any, out: Any | None = None) -> Any:
        return self.executable.run(value, out=out)


def compile(
    program: CommunicationProgram,
    context: CompileContext,
    *,
    bindings: RuntimeBindings,
    registry: BackendRegistry,
    evidence: BenchmarkEvidence | EvidenceCatalog | None = None,
) -> BoundExecutable:
    verify(program, context)
    target = context.device_type
    backend = registry.resolve(target)
    capabilities = backend.capabilities(context)
    effective, fallback_reason, evidence_id = _select_effective_program(
        program,
        context,
        target,
        capabilities,
        evidence,
    )
    _require_supported(effective, capabilities)
    lowered = backend.lower(effective, context, bindings)
    _require_preferred_primitive(context, lowered)
    executable = backend.compile(lowered)
    info = ExecutionInfo(
        requested_algorithm=_algorithm_name(program.algorithm),
        effective_algorithm=_algorithm_name(effective.algorithm),
        requested_wire=program.wire,
        effective_wire=effective.wire,
        physical_primitive=lowered.physical_primitive.value,
        requested_output=_output_name(program.output),
        effective_output=_output_name(effective.output),
        logical_bytes=_logical_bytes(context),
        estimated_wire_bytes=_estimated_wire_bytes(effective, context),
        fused_stages=tuple(stage.name for stage in lowered.stages),
        workspace_bytes=lowered.buffer_plan.total_bytes,
        topology_signature=context.topology_signature,
        world_size=context.world_size,
        fallback_reason=fallback_reason,
        evidence_id=evidence_id,
    )
    return BoundExecutable(executable, lowered, info)


def _select_effective_program(
    program: CommunicationProgram,
    context: CompileContext,
    target: str,
    capabilities: BackendCapabilities,
    evidence: BenchmarkEvidence | None,
) -> tuple[CommunicationProgram, str | None, str | None]:
    if not isinstance(program.algorithm, AutoAlgorithm):
        return program, None, None

    selected_evidence = (
        evidence.match(target, context, program, capabilities)
        if isinstance(evidence, EvidenceCatalog)
        else evidence
    )
    decision = decide_auto(
        target,
        context,
        program,
        capabilities,
        selected_evidence,
    )
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
    requested = _capability_key(program)
    for specification in capabilities.specifications:
        candidate = (
            specification.operation,
            specification.output,
            specification.wire,
            specification.algorithm,
            specification.dtype,
            specification.bit,
            specification.group_size,
            specification.quant_type,
            specification.compact,
        )
        if candidate == requested and (
            not program.async_op or specification.async_supported
        ):
            return
    wire = program.wire
    detail = (
        f"{wire.bit}-bit group_size={wire.group_size} "
        f"quant_type={wire.quant_type} compact={wire.compact}"
        if isinstance(wire, QuantizedWire)
        else f"full-precision dtype={wire.dtype.value}"
    )
    raise UnsupportedProgram(
        f"backend {capabilities.target!r} does not support "
        f"{_operation_name(program.operation)} {_output_name(program.output)} "
        f"{_algorithm_name(program.algorithm)} with {detail}"
    )


def _capability_key(program: CommunicationProgram) -> tuple[object, ...]:
    wire = program.wire
    return (
        _operation_name(program.operation),
        _output_name(program.output),
        "quantized" if isinstance(wire, QuantizedWire) else "full_precision",
        _algorithm_name(program.algorithm),
        program.output.dtype,
        wire.bit if isinstance(wire, QuantizedWire) else None,
        wire.group_size if isinstance(wire, QuantizedWire) else None,
        wire.quant_type if isinstance(wire, QuantizedWire) else None,
        wire.compact if isinstance(wire, QuantizedWire) else None,
    )


def _operation_name(operation: object) -> str:
    if isinstance(operation, ReduceSum):
        return "sum"
    if isinstance(operation, ReduceMean):
        return "mean"
    raise UnsupportedProgram(f"unknown operation type {type(operation).__name__}")


def _output_name(output: object) -> str:
    if isinstance(output, FullTensor):
        return "full_tensor"
    if isinstance(output, ReducedShard):
        return "reduced_shard"
    raise UnsupportedProgram(f"unknown output type {type(output).__name__}")


def _algorithm_name(algorithm: object) -> str:
    try:
        return _ALGORITHM_NAMES[type(algorithm)]
    except KeyError as error:
        raise UnsupportedProgram(
            f"unknown algorithm type {type(algorithm).__name__}"
        ) from error


def _require_preferred_primitive(
    context: CompileContext,
    lowered: LoweredProgram,
) -> None:
    preferred = context.preferred_primitive
    if preferred is not None and lowered.physical_primitive is not preferred:
        raise UnsupportedProgram(
            f"backend {lowered.target!r} lowered "
            f"{lowered.physical_primitive.value!r}, not requested "
            f"{preferred.value!r}"
        )


def _logical_bytes(context: CompileContext) -> int:
    numel = 1
    for size in context.shape:
        numel *= size
    element_bytes = {
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
    }[context.dtype.value]
    return numel * element_bytes


def _estimated_wire_bytes(
    program: CommunicationProgram,
    context: CompileContext,
) -> int:
    logical_bytes = _logical_bytes(context)
    wire = program.wire
    algorithm = program.algorithm
    if isinstance(wire, FullPrecisionWire):
        return 2 * logical_bytes * (context.world_size - 1) // context.world_size
    numel = 1
    for size in context.shape:
        numel *= size
    groups = (numel + wire.group_size - 1) // wire.group_size
    scale_bytes = 4 if context.dtype.value == "fp32" else 2
    payload = groups * (wire.group_size * wire.bit // 8 + scale_bytes)
    if isinstance(algorithm, CompressedAllGather):
        return payload * (context.world_size - 1)
    if isinstance(algorithm, CompressedReduceScatter):
        return payload * (context.world_size - 1) // context.world_size
    if isinstance(algorithm, CompressedReduceScatterAllGather):
        return 2 * payload * (context.world_size - 1) // context.world_size
    return logical_bytes
