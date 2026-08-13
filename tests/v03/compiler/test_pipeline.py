from __future__ import annotations

import pytest

from lowbit_comm.backends.reference import ReferenceBackend
from lowbit_comm.compiler import BackendRegistry, UnsupportedProgram, compile
from lowbit_comm.core import (
    BackendCapabilities,
    CapabilitySpec,
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
            specifications=tuple(
                spec for spec in capabilities.specifications if spec.bit in {None, 8}
            ),
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


def test_capability_match_rejects_unsupported_compact_fulltensor_before_lowering() -> None:
    from lowbit_comm.backends.cuda import CudaBackend
    from lowbit_comm.backends.cuda.loader import CudaExtensionStatus

    registry = BackendRegistry()
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(True, object(), abi_version=1)
    )
    registry.register("cuda", backend)
    context = CompileContext(
        rank=0,
        world_size=2,
        shape=(64,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
    )
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64, compact=True),
        algorithm=CompressedReduceScatterAllGather(),
    )

    with pytest.raises(UnsupportedProgram, match="compact=True"):
        compile(
            program,
            context,
            bindings=RuntimeBindings(),
            registry=registry,
        )


def test_capability_spec_is_a_combined_program_contract() -> None:
    spec = CapabilitySpec(
        operation="mean",
        output="full_tensor",
        wire="quantized",
        algorithm="compressed_rs_ag",
        dtype=DataType.FP16,
        bit=8,
        group_size=64,
        quant_type="linear",
        compact=False,
        async_supported=True,
        physical_primitive="all_to_all_local_reduce",
    )

    assert spec.operation == "mean"
    assert spec.output == "full_tensor"
    assert spec.bit == 8


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
