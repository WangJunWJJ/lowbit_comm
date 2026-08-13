from __future__ import annotations

from lowbit_comm.backends.reference import ReferenceBackend
import pytest

from lowbit_comm.compiler import (
    BackendRegistry,
    BenchmarkEvidence,
    EvidenceCatalog,
    compile,
)
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
        device_architecture="sm_86",
        topology_signature="single_node_pcie",
        node_count=1,
        software_fingerprint="torch2.4-cuda12.1-nccl2.20",
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


def _evidence(**overrides: object) -> BenchmarkEvidence:
    values = {
        "evidence_id": "a6000-4rank-v1",
        "target": "reference",
        "device_architecture": "sm_86",
        "topology_signature": "single_node_pcie",
        "world_size": 4,
        "node_count": 1,
        "software_fingerprint": "torch2.4-cuda12.1-nccl2.20",
        "shape": (4096,),
        "dtype": DataType.FP16,
        "operation": "mean",
        "output": "full_tensor",
        "algorithm": "compressed_rs_ag",
        "physical_primitive": "reference_rs_ag",
        "bit": 8,
        "group_size": 64,
        "quant_type": "linear",
        "compact": True,
        "speedup_percent": 7.5,
    }
    values.update(overrides)
    return BenchmarkEvidence(**values)


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
    evidence = _evidence()
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
    evidence = _evidence(
        evidence_id="wrong-topology",
        topology_signature="dual_node_tcp",
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


def test_auto_rejects_evidence_from_different_device_or_software() -> None:
    for evidence in (
        _evidence(device_architecture="sm_89"),
        _evidence(software_fingerprint="torch2.5-cuda12.4-nccl2.21"),
        _evidence(node_count=2),
    ):
        executable = compile(
            _program(),
            _context(),
            bindings=RuntimeBindings(),
            registry=_registry(),
            evidence=evidence,
        )

        assert executable.execution_info.effective_algorithm == "native"
        assert executable.execution_info.fallback_reason == "benchmark evidence mismatch"


def test_auto_rejects_evidence_from_different_workload_or_wire() -> None:
    for evidence in (
        _evidence(shape=(8192,)),
        _evidence(dtype=DataType.BF16),
        _evidence(bit=4),
        _evidence(group_size=32),
        _evidence(compact=False),
    ):
        executable = compile(
            _program(),
            _context(),
            bindings=RuntimeBindings(),
            registry=_registry(),
            evidence=evidence,
        )

        assert executable.execution_info.effective_algorithm == "native"


def test_auto_rejects_unavailable_physical_primitive_evidence() -> None:
    executable = compile(
        _program(),
        _context(),
        bindings=RuntimeBindings(),
        registry=_registry(),
        evidence=_evidence(physical_primitive="hierarchical_reduce_scatter"),
    )

    assert executable.execution_info.effective_algorithm == "native"
    assert executable.execution_info.fallback_reason == "benchmark evidence mismatch"


def test_auto_selects_exact_evidence_from_catalog() -> None:
    catalog = EvidenceCatalog(
        (
            _evidence(evidence_id="stale", topology_signature="dual_node_tcp"),
            _evidence(evidence_id="exact", speedup_percent=12.5),
        )
    )

    executable = compile(
        _program(),
        _context(),
        bindings=RuntimeBindings(),
        registry=_registry(),
        evidence=catalog,
    )

    assert executable.execution_info.effective_algorithm == "compressed_rs_ag"
    assert executable.execution_info.evidence_id == "exact"


def test_evidence_catalog_rejects_duplicate_ids_and_conflicting_fingerprints() -> None:
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        EvidenceCatalog((_evidence(), _evidence()))

    with pytest.raises(ValueError, match="conflicting benchmark evidence"):
        EvidenceCatalog(
            (
                _evidence(evidence_id="first", speedup_percent=5.0),
                _evidence(evidence_id="second", speedup_percent=15.0),
            )
        )
