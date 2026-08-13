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
    HierarchicalCompressed,
    QuantizedWire,
    ReduceMean,
    RuntimeBindings,
    PhysicalPrimitive,
    LoweredStage,
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


def test_bound_executable_forwards_caller_owned_output() -> None:
    class Executable:
        def __init__(self) -> None:
            self.received: tuple[object, object] | None = None

        def run(self, value: object, out: object | None = None) -> str:
            self.received = (value, out)
            return "work"

    from lowbit_comm.compiler.pipeline import BoundExecutable
    from lowbit_comm.core import ExecutionInfo

    inner = Executable()
    lowered = object()
    bound = BoundExecutable(
        inner,
        lowered,  # type: ignore[arg-type]
        ExecutionInfo(
            requested_algorithm="native",
            effective_algorithm="native",
            requested_wire=object(),
            effective_wire=object(),
            physical_primitive="reference",
        ),
    )
    value = object()
    output = object()

    assert bound.run(value, out=output) == "work"
    assert inner.received == (value, output)


def test_execution_info_reports_effective_physical_primitive() -> None:
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

    assert executable.execution_info.physical_primitive == "reference_rs_ag"
    assert executable.lowered.physical_primitive.value == "reference_rs_ag"


def test_execution_info_reports_output_bytes_fusion_and_workspace() -> None:
    registry = BackendRegistry()
    registry.register("reference", ReferenceBackend())

    executable = compile(
        CommunicationProgram(
            operation=ReduceMean(),
            output=FullTensor(DataType.FP16),
            wire=QuantizedWire(bit=8, group_size=64),
            algorithm=CompressedReduceScatterAllGather(),
        ),
        _context(),
        bindings=RuntimeBindings(),
        registry=registry,
    )
    info = executable.execution_info

    assert info.requested_output == "full_tensor"
    assert info.effective_output == "full_tensor"
    assert info.logical_bytes == 4
    assert info.estimated_wire_bytes > 0
    assert info.fused_stages == tuple(stage.name for stage in executable.lowered.stages)
    assert info.workspace_bytes == executable.lowered.buffer_plan.total_bytes
    assert info.topology_signature == "unknown"
    assert info.world_size == 2


def test_explicit_physical_primitive_must_match_lowered_implementation() -> None:
    registry = BackendRegistry()
    registry.register("reference", ReferenceBackend())
    context = CompileContext(
        rank=0,
        world_size=2,
        shape=(2,),
        dtype=DataType.FP16,
        device_type="reference",
        preferred_primitive=PhysicalPrimitive.RING_REDUCE_SCATTER,
    )
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    with pytest.raises(UnsupportedProgram, match="ring_reduce_scatter"):
        compile(
            program,
            context,
            bindings=RuntimeBindings(),
            registry=registry,
        )


def test_matching_explicit_physical_primitive_compiles() -> None:
    registry = BackendRegistry()
    registry.register("reference", ReferenceBackend())
    context = CompileContext(
        rank=0,
        world_size=2,
        shape=(2,),
        dtype=DataType.FP16,
        device_type="reference",
        preferred_primitive=PhysicalPrimitive.REFERENCE_RS_AG,
    )
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(bit=8, group_size=64),
        algorithm=CompressedReduceScatterAllGather(),
    )

    executable = compile(
        program,
        context,
        bindings=RuntimeBindings(),
        registry=registry,
    )

    assert executable.execution_info.physical_primitive == "reference_rs_ag"


def test_hierarchical_wire_estimate_uses_grouped_schedule() -> None:
    from lowbit_comm.backends.cuda import CudaBackend
    from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
    from lowbit_comm.backends.cuda.transports import GroupedTransportRuntime

    module = type(
        "Module",
        (),
        {
            "inplace_quantize": staticmethod(lambda *args: None),
            "inplace_dequantize": staticmethod(lambda *args: None),
            "inplace_quantize_chunks": staticmethod(lambda *args: True),
            "inplace_dequantize_reduce_mean_requantize": staticmethod(
                lambda *args: True
            ),
            "QuantType": type("QuantType", (), {"Linear": object()}),
            "DType": type("DType", (), {"FP16": object()}),
        },
    )()
    registry = BackendRegistry()
    registry.register(
        "cuda",
        CudaBackend(extension_status=CudaExtensionStatus(True, module, abi_version=1)),
    )
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=0,0,1,1",
        node_count=2,
    )
    executable = compile(
        CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            QuantizedWire(8, 64, compact=False),
            HierarchicalCompressed(max_fan_in=8),
        ),
        context,
        bindings=RuntimeBindings(
            backend_runtime=GroupedTransportRuntime(
                new_group=lambda ranks: tuple(ranks)
            )
        ),
        registry=registry,
    )
    payload_bytes = 64 * (64 + 2)

    assert executable.execution_info.estimated_wire_bytes == 4 * payload_bytes
    assert executable.execution_info.estimated_wire_bytes < (
        executable.execution_info.logical_bytes * context.world_size
    )


def test_hierarchical_wire_estimate_accounts_for_group_peers() -> None:
    from lowbit_comm.compiler.pipeline import _estimated_wire_bytes
    from lowbit_comm.core import (
        GroupedReductionPlan,
        LoweredProgram,
        ExecutorKind,
        PhysicalPrimitive,
        ReduceMean,
        compile_reduction,
    )

    context = CompileContext(
        rank=0,
        world_size=8,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=8),
    )
    lowered = LoweredProgram(
        target="cuda",
        program=program,
        stages=(
            LoweredStage("group", program.wire),
        ),
        reduction=compile_reduction(program.operation, context.world_size),
        context=context,
        bindings=RuntimeBindings(),
        executor_kind=ExecutorKind.HIERARCHICAL_COMPRESSED,
        physical_primitive=PhysicalPrimitive.HIERARCHICAL_COMPRESSED_FULL_TENSOR,
        grouped_reduction=GroupedReductionPlan(
            world_size=8,
            max_fan_in=8,
            levels=(((0, 1, 2, 3, 4, 5, 6, 7),),),
        ),
    )
    payload_bytes = 64 * (64 + 2)

    assert _estimated_wire_bytes(program, context, lowered) == (
        2 * 7 * payload_bytes
    )
