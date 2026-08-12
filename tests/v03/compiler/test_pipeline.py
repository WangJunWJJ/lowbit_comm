from __future__ import annotations

import pytest

from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler import BackendRegistry, UnsupportedProgram, compile
from lowbit_comm.core import (
    BackendCapabilities,
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
)


class Int8OnlyReferenceBackend(ReferenceBackend):
    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        capabilities = super().capabilities(context)
        return BackendCapabilities(
            target=capabilities.target,
            supported_bits=frozenset({8}),
            supported_algorithms=capabilities.supported_algorithms,
        )


def _context() -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=2,
        shape=(2,),
        dtype=DataType.FP16,
        device_type="reference",
    )


def test_explicit_unsupported_program_fails_instead_of_falling_back() -> None:
    registry = BackendRegistry()
    registry.register("reference", Int8OnlyReferenceBackend())
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=4, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    with pytest.raises(UnsupportedProgram, match="4-bit"):
        compile(
            program,
            _context(),
            bindings=RuntimeBindings(),
            registry=registry,
        )


def test_compiled_executable_does_not_revisit_registry() -> None:
    registry = BackendRegistry()
    registry.register("reference", ReferenceBackend())
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )
    executable = compile(
        program,
        _context(),
        bindings=RuntimeBindings(),
        registry=registry,
    )

    registry.resolve = lambda target: (_ for _ in ()).throw(AssertionError(target))

    assert executable.run([1.0, 2.0]).wait() == [1.0, 2.0]
