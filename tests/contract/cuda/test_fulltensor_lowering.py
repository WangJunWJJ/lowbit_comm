from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.backends.cuda.transports import (
    GroupedTransportRuntime,
    bind_grouped_transport,
)
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatterAllGather,
    CompressedAllGather,
    DataType,
    FullTensor,
    QuantizedWire,
    ReduceMean,
    ReduceSum,
    RuntimeBindings,
    WorkspaceRole,
    NativeAllReduce,
    FullPrecisionWire,
    HierarchicalCompressed,
    ExecutorKind,
    LoweredProgram,
)


def _program(*, compact: bool = False) -> CommunicationProgram:
    return CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=compact),
        algorithm=CompressedReduceScatterAllGather(),
    )


def _context() -> CompileContext:
    return CompileContext(
        rank=0,
        world_size=4,
        shape=(131_073,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
    )


def _native() -> object:
    return SimpleNamespace(
        inplace_quantize=lambda *args: None,
        inplace_quantize_chunks=lambda *args: True,
        inplace_dequantize_reduce_mean=lambda *args: True,
        inplace_dequantize_reduce_mean_requantize=lambda *args: True,
        inplace_dequantize_gathered=lambda *args: True,
        QuantType=SimpleNamespace(Linear=object()),
        DType=SimpleNamespace(FP16=object()),
    )


def test_fulltensor_lowering_has_exactly_two_quantized_collectives() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    collective_stages = [stage for stage in lowered.stages if stage.collective]
    assert [stage.name for stage in collective_stages] == [
        "quantized_reduce_scatter",
        "quantized_all_gather",
    ]
    assert all(stage.wire == _program().wire for stage in collective_stages)
    assert [stage.name for stage in lowered.stages][-1] == "gathered_dequant_writeback"


def test_hierarchical_fulltensor_lowering_freezes_grouped_schedule() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(131_073,),
        dtype=DataType.FP16,
        device_type="cuda",
        device_architecture="sm86",
        topology_signature="node_ids=0,0,1,1",
        node_count=2,
    )
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=False),
        algorithm=HierarchicalCompressed(max_fan_in=8),
    )

    lowered = backend.lower(program, context, RuntimeBindings())

    assert lowered.executor_kind is ExecutorKind.HIERARCHICAL_COMPRESSED
    assert lowered.physical_primitive.value == "hierarchical_compressed_full_tensor"
    assert lowered.grouped_reduction is not None
    assert lowered.grouped_reduction.levels == (
        ((0, 1), (2, 3)),
        ((0, 2),),
    )
    assert [stage.name for stage in lowered.stages] == [
        "quantized_group_reduce_level_0",
        "quantized_group_reduce_level_1",
        "normalize_and_requantize_root",
        "quantized_group_broadcast_level_1",
        "quantized_group_broadcast_level_0",
        "gathered_dequant_writeback",
    ]


def test_hierarchical_compile_requires_prebound_process_groups() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=0,0,1,1",
        node_count=2,
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=8),
    )
    lowered = backend.lower(program, context, RuntimeBindings())

    with pytest.raises(ValueError, match="GroupedTransportBindings"):
        backend.compile(lowered)

    assert lowered.grouped_reduction is not None
    bindings = bind_grouped_transport(
        lowered.grouped_reduction,
        new_group=lambda ranks: tuple(ranks),
    )
    lowered = backend.lower(
        program,
        context,
        RuntimeBindings(backend_runtime=bindings),
    )

    executable = backend.compile(lowered)
    assert type(executable).__name__ == "CudaHierarchicalFullTensorExecutable"
    assert callable(executable.run)


def test_hierarchical_compile_binds_groups_once_from_runtime_factory() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=0,0,1,1",
        node_count=2,
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=8),
    )
    calls: list[tuple[int, ...]] = []
    runtime = GroupedTransportRuntime(
        new_group=lambda ranks: calls.append(tuple(ranks)) or tuple(ranks)
    )

    lowered = backend.lower(
        program,
        context,
        RuntimeBindings(backend_runtime=runtime),
    )
    executable = backend.compile(lowered)

    assert calls == [(0, 1), (2, 3), (0, 2)]
    assert type(executable).__name__ == "CudaHierarchicalFullTensorExecutable"
    assert lowered.bindings.backend_runtime is runtime
    assert executable.lowered.bindings.backend_runtime is not runtime


def test_hierarchical_workspace_is_bounded_by_max_group_fan_in() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=64,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=" + ",".join(str(rank // 8) for rank in range(64)),
        node_count=8,
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=8),
    )

    lowered = backend.lower(program, context, RuntimeBindings())
    receive = next(
        item for item in lowered.buffer_plan.buffers if item.role is WorkspaceRole.RECEIVE
    )

    assert receive.shape[0] == 8
    assert receive.shape[0] < context.world_size


def test_hierarchical_lowering_rejects_kernel_fan_in_above_eight() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=16,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=" + ",".join("0" for _ in range(16)),
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=16),
    )

    with pytest.raises(ValueError, match="max_fan_in.*8"):
        backend.lower(program, context, RuntimeBindings())


def test_hierarchical_singleton_groups_do_not_launch_global_collectives() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    context = CompileContext(
        rank=0,
        world_size=3,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids=0,1,1",
        node_count=2,
    )
    program = CommunicationProgram(
        ReduceMean(),
        FullTensor(DataType.FP16),
        QuantizedWire(8, 64, compact=False),
        HierarchicalCompressed(max_fan_in=2),
    )
    lowered = backend.lower(program, context, RuntimeBindings())
    assert lowered.grouped_reduction is not None
    bindings = bind_grouped_transport(
        lowered.grouped_reduction,
        new_group=lambda ranks: tuple(ranks),
    )
    lowered = backend.lower(
        program,
        context,
        RuntimeBindings(backend_runtime=bindings),
    )
    executable = backend.compile(lowered)

    participated = executable._participating_groups()  # noqa: SLF001

    assert (0,) not in participated
    assert participated == ((0, 1),)


def test_fulltensor_compile_rejects_unfused_compact_wire() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    lowered = backend.lower(_program(compact=True), _context(), RuntimeBindings())

    with pytest.raises(RuntimeError, match="compact=False"):
        backend.compile(lowered)


def test_cuda_capabilities_do_not_advertise_unexecutable_fulltensor_wires() -> None:
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(True, _native(), abi_version=1)
    )

    capabilities = backend.capabilities(_context())
    fulltensor_specs = [
        spec
        for spec in capabilities.specifications
        if spec.algorithm == "compressed_rs_ag"
    ]

    assert fulltensor_specs
    assert {spec.bit for spec in fulltensor_specs} == {8}
    assert {spec.group_size for spec in fulltensor_specs} == {64}
    assert {spec.compact for spec in fulltensor_specs} == {False}


def test_cuda_capabilities_hide_compression_on_extension_abi_mismatch() -> None:
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(True, _native(), abi_version=99)
    )

    capabilities = backend.capabilities(_context())

    assert {spec.algorithm for spec in capabilities.specifications} == {"native"}


def test_fulltensor_compile_requires_both_fused_native_symbols() -> None:
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(
            True,
                SimpleNamespace(
                    inplace_quantize=lambda *args: None,
                    inplace_quantize_chunks=lambda *args: True,
                inplace_dequantize_reduce_mean_requantize=lambda *args: True,
                QuantType=SimpleNamespace(Linear=object()),
                DType=SimpleNamespace(FP16=object()),
            ),
        )
    )
    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    with pytest.raises(RuntimeError, match="inplace_dequantize_gathered"):
        backend.compile(lowered)


def test_fulltensor_query_checks_completion_before_advancing_pipeline() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")
    work_body = source.split("class _TwoCollectiveWork:", 1)[1].split(
        "def _require_module", 1
    )[0]

    query_body = work_body.split("def query", 1)[1].split("def wait", 1)[0]
    assert "_handle_query(self._first)" in query_body
    assert "_handle_query(self._second)" in query_body


def test_fulltensor_completion_avoids_device_wide_synchronize() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")

    assert "torch.cuda.synchronize(" not in source
    assert "event.record(" in source
    assert "event.query()" in source


def test_cuda_executors_do_not_use_host_thread_pool_or_busy_spin() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in source
    assert "sleep(0)" not in source


def test_cuda_executors_lease_compiled_internal_workspaces() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "executors.py"
    ).read_text(encoding="utf-8")

    assert "CudaWorkspaceManager" in source
    assert "_workspace.acquire" in source


def test_rsag_executable_reuses_send_payload_for_feedback() -> None:
    source = (
        Path(__file__).parents[3]
        / "src/lowbit_comm/backends/cuda/executors.py"
    ).read_text(encoding="utf-8")
    body = source.split("class CudaFullTensorExecutable:", 1)[1]

    assert "def run_with_local_reconstruction(" in body
    assert "send[destination, : self.payload_numel]" in body


def test_rsag_uses_one_chunk_quantization_dispatch() -> None:
    source = (
        Path(__file__).parents[3] / "src/lowbit_comm/backends/cuda/executors.py"
    ).read_text(encoding="utf-8")
    body = source.split("class CudaFullTensorExecutable:", 1)[1].split(
        "class CudaHierarchicalFullTensorExecutable:", 1
    )[0]

    assert "quantize_chunks_into(" in body
    assert "for destination in range(self.plan.world_size):\n            quantize_into(" not in body


def test_cuda_backend_compiles_explicit_native_all_reduce() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(False, None, "unused"))
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=FullPrecisionWire(DataType.FP16),
        algorithm=NativeAllReduce(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings(process_group="group"))
    executable = backend.compile(lowered)

    assert [stage.name for stage in lowered.stages] == ["native_all_reduce"]
    assert type(executable).__name__ == "CudaNativeAllReduceExecutable"


def test_cuda_backend_compiles_explicit_compressed_all_gather() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64),
        algorithm=CompressedAllGather(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert [stage.name for stage in lowered.stages] == [
        "quantize_full_contribution",
        "compressed_all_gather",
        "fused_dequant_reduce_mean",
    ]
    executable = backend.compile(lowered)
    assert type(executable).__name__ == "CudaCompressedAllGatherExecutable"
    assert callable(executable.reconstruct_local)


@pytest.mark.parametrize(
    ("operation", "expected_divisor", "stage_suffix"),
    [(ReduceSum(), 1, "sum"), (ReduceMean(), 4, "mean")],
)
def test_fulltensor_lowering_preserves_reduction_semantics(
    operation: object,
    expected_divisor: int,
    stage_suffix: str,
) -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))
    program = CommunicationProgram(
        operation=operation,
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=False),
        algorithm=CompressedReduceScatterAllGather(),
    )

    lowered = backend.lower(program, _context(), RuntimeBindings())

    assert lowered.reduction.divisor == expected_divisor
    assert f"reduce_{stage_suffix}" in lowered.stages[2].name


def test_cuda_compile_uses_lowered_executor_kind_as_single_authority() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(False, None, "unused"))
    native_program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=FullPrecisionWire(DataType.FP16),
        algorithm=NativeAllReduce(),
    )
    lowered = backend.lower(native_program, _context(), RuntimeBindings())
    conflicting_program = CommunicationProgram(
        operation=ReduceMean(),
        output=FullTensor(DataType.FP16),
        wire=QuantizedWire(8, 64, compact=False),
        algorithm=CompressedReduceScatterAllGather(),
    )
    lowered = LoweredProgram(
        target=lowered.target,
        program=conflicting_program,
        stages=lowered.stages,
        executor_kind=ExecutorKind.NATIVE_ALL_REDUCE,
        reduction=lowered.reduction,
        context=lowered.context,
        bindings=lowered.bindings,
        physical_primitive=lowered.physical_primitive,
    )

    executable = backend.compile(lowered)

    assert type(executable).__name__ == "CudaNativeAllReduceExecutable"


def test_lowered_stages_express_primitives_dependencies_and_stream_roles() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))

    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    assert lowered.stages[0].primitive == "quantize_destination_chunks"
    assert lowered.stages[1].dependencies == (lowered.stages[0].stage_id,)
    assert lowered.stages[1].stream_role == "communication"
    assert lowered.stages[-1].stream_role == "compute"


def test_fulltensor_lowering_plans_only_internal_reusable_workspaces() -> None:
    backend = CudaBackend(extension_status=CudaExtensionStatus(True, _native()))

    lowered = backend.lower(_program(), _context(), RuntimeBindings())

    roles = {buffer.role for buffer in lowered.buffer_plan.buffers}
    assert roles == {
        WorkspaceRole.PADDED_INPUT,
        WorkspaceRole.SEND,
        WorkspaceRole.RECEIVE,
        WorkspaceRole.REDUCED_PAYLOAD,
        WorkspaceRole.GATHERED_PAYLOAD,
        WorkspaceRole.LOCAL_RECONSTRUCTION,
        WorkspaceRole.RESTORED_SCRATCH,
    }
    assert WorkspaceRole.OUTPUT not in roles
