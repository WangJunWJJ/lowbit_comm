from __future__ import annotations

from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler import BackendRegistry, BenchmarkEvidence, compile
from lowbit_comm.core import (
    AutoAlgorithm,
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    DataType,
    FullPrecisionWire,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


def _context() -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="reference",
        topology_signature="single_node_pcie",
    )


def _program() -> CommunicationProgram:
    return CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=AutoAlgorithm(),
    )


def _registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("reference", ReferenceBackend())
    return registry


def test_auto_without_evidence_compiles_observable_native_fallback() -> None:
    executable = compile(
        _program(),
        _context(),
        bindings=RuntimeBindings(),
        registry=_registry(),
    )

    assert executable.execution_info.requested_algorithm == "auto"
    assert executable.execution_info.effective_algorithm == "native"
    assert isinstance(executable.execution_info.effective_wire, FullPrecisionWire)
    assert executable.execution_info.fallback_reason == "missing benchmark evidence"


def test_auto_uses_matching_positive_evidence_for_compression() -> None:
    evidence = BenchmarkEvidence(
        evidence_id="a6000-4rank-v1",
        target="reference",
        topology_signature="single_node_pcie",
        world_size=4,
        speedup_percent=7.5,
    )
    executable = compile(
        _program(),
        _context(),
        bindings=RuntimeBindings(),
        registry=_registry(),
        evidence=evidence,
    )

    assert executable.execution_info.effective_algorithm == "compressed_rs_ag"
    assert isinstance(executable.lowered.program.algorithm, CompressedReduceScatterAllGather)
    assert executable.execution_info.evidence_id == "a6000-4rank-v1"


def test_auto_rejects_stale_topology_evidence() -> None:
    evidence = BenchmarkEvidence(
        evidence_id="wrong-topology",
        target="reference",
        topology_signature="dual_node_tcp",
        world_size=4,
        speedup_percent=20.0,
    )
    executable = compile(
        _program(),
        _context(),
        bindings=RuntimeBindings(),
        registry=_registry(),
        evidence=evidence,
    )

    assert executable.execution_info.effective_algorithm == "native"
    assert executable.execution_info.fallback_reason == "benchmark evidence mismatch"
