"""Evidence-gated compile-time strategy selection."""

from __future__ import annotations

from dataclasses import dataclass

from lowbit_comm.core.backend import BackendCapabilities
from lowbit_comm.core.context import CompileContext
from lowbit_comm.core.operations import ReduceMean, ReduceSum
from lowbit_comm.core.program import CommunicationProgram
from lowbit_comm.core.types import DataType, FullTensor, QuantizedWire, ReducedShard


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    evidence_id: str
    target: str
    device_architecture: str
    topology_signature: str
    world_size: int
    node_count: int
    software_fingerprint: str
    shape: tuple[int, ...]
    dtype: DataType
    operation: str
    output: str
    algorithm: str
    physical_primitive: str
    bit: int
    group_size: int
    quant_type: str
    compact: bool
    speedup_percent: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported benchmark evidence schema")
        if not isinstance(self.dtype, DataType):
            raise TypeError("evidence dtype must be a DataType")
        if self.world_size <= 0 or self.node_count <= 0:
            raise ValueError("evidence rank and node counts must be positive")
        object.__setattr__(self, "shape", tuple(self.shape))

    def matches(
        self,
        target: str,
        context: CompileContext,
        program: CommunicationProgram,
        capabilities: BackendCapabilities,
    ) -> bool:
        wire = program.wire
        if not isinstance(wire, QuantizedWire):
            return False
        candidate = next(
            (
                spec
                for spec in capabilities.specifications
                if spec.operation == self.operation
                and spec.output == self.output
                and spec.wire == "quantized"
                and spec.algorithm == self.algorithm
                and spec.physical_primitive == self.physical_primitive
                and spec.dtype == self.dtype
                and spec.bit == self.bit
                and spec.group_size == self.group_size
                and spec.quant_type == self.quant_type
                and spec.compact == self.compact
            ),
            None,
        )
        return (
            self.target == target
            and self.device_architecture == context.device_architecture
            and self.topology_signature == context.topology_signature
            and self.world_size == context.world_size
            and self.node_count == context.node_count
            and self.software_fingerprint == context.software_fingerprint
            and tuple(self.shape) == context.shape
            and self.dtype == context.dtype
            and self.operation == _operation_name(program)
            and self.output == _output_name(program)
            and self.algorithm == "compressed_rs_ag"
            and self.bit == wire.bit
            and self.group_size == wire.group_size
            and self.quant_type == wire.quant_type
            and self.compact == wire.compact
            and candidate is not None
        )


@dataclass(frozen=True, slots=True)
class AutoDecision:
    use_compression: bool
    fallback_reason: str | None
    evidence_id: str | None


def decide_auto(
    target: str,
    context: CompileContext,
    program: CommunicationProgram,
    capabilities: BackendCapabilities,
    evidence: BenchmarkEvidence | None,
) -> AutoDecision:
    if evidence is None:
        return AutoDecision(False, "missing benchmark evidence", None)
    if not evidence.matches(target, context, program, capabilities):
        return AutoDecision(False, "benchmark evidence mismatch", None)
    if evidence.speedup_percent <= 0:
        return AutoDecision(False, "benchmark evidence shows no speedup", evidence.evidence_id)
    return AutoDecision(True, None, evidence.evidence_id)


def _operation_name(program: CommunicationProgram) -> str:
    if isinstance(program.operation, ReduceMean):
        return "mean"
    if isinstance(program.operation, ReduceSum):
        return "sum"
    return "unknown"


def _output_name(program: CommunicationProgram) -> str:
    if isinstance(program.output, FullTensor):
        return "full_tensor"
    if isinstance(program.output, ReducedShard):
        return "reduced_shard"
    return "unknown"
