from __future__ import annotations

import pytest

from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler import BackendRegistry
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


def _program() -> CommunicationProgram:
    return CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )


def _context() -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=2,
        shape=(2,),
        dtype=DataType.FP16,
        device_type="cpu",
    )


def test_registry_resolves_backends_by_target_only() -> None:
    registry = BackendRegistry()
    backend = ReferenceBackend()
    registry.register("reference", backend)

    assert registry.resolve("reference") is backend
    with pytest.raises(KeyError, match="already registered"):
        registry.register("reference", ReferenceBackend())
    with pytest.raises(KeyError, match="unknown backend target"):
        registry.resolve("cuda:sm86")


def test_reference_backend_lowers_observable_wire_stages() -> None:
    backend = ReferenceBackend()
    program = _program()
    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert lowered.program == program
    assert [stage.name for stage in lowered.stages] == [
        "quantized_reduce_scatter",
        "quantized_all_gather",
    ]
    assert all(stage.wire == program.wire for stage in lowered.stages)


def test_reference_executable_returns_unified_work() -> None:
    backend = ReferenceBackend()
    lowered = backend.lower(_program(), _context(), RuntimeBindings())
    executable = backend.compile(lowered)

    work = executable.run([1.0, 2.0])

    assert work.query() is True
    assert work.wait() == [1.0, 2.0]
    assert work.query() is True


def test_capabilities_are_immutable_and_target_specific() -> None:
    backend = ReferenceBackend()
    capabilities = backend.capabilities(_context())

    assert capabilities.target == "reference"
    assert capabilities.supported_bits == frozenset({4, 8})
    with pytest.raises(AttributeError):
        capabilities.target = "cuda"
