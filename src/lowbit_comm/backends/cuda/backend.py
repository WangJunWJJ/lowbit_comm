"""CUDA Backend Protocol implementation."""

from __future__ import annotations

from dataclasses import replace

from lowbit_comm.core import (
    BackendCapabilities,
    CapabilitySpec,
    BufferPlan,
    BufferSpec,
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    HierarchicalCompressed,
    RuntimeBindings,
    NativeAllReduce,
    compile_reduction,
    DataType,
    WorkspaceRole,
)
from lowbit_comm.core.lowered import (
    ExecutorKind,
    LoweredProgram,
    LoweredStage,
    StageKind,
)
from lowbit_comm.core.primitives import PhysicalPrimitive
from lowbit_comm.core.topology import (
    compile_grouped_reduction,
    parse_topology_signature,
)

from .executors import (
    CudaCompressedAllGatherExecutable,
    CudaFullTensorExecutable,
    CudaHierarchicalFullTensorExecutable,
    CudaNativeAllReduceExecutable,
    CudaReducedShardExecutable,
)
from .loader import CudaExtensionStatus, load_cuda_extension
from .transports import (
    GroupedTransportBindings,
    GroupedTransportRuntime,
    bind_grouped_transport,
)


class CudaBackend:
    name = "cuda"
    abi_version = 1

    def __init__(self, *, extension_status: CudaExtensionStatus | None = None) -> None:
        self._status = extension_status or load_cuda_extension()

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        return BackendCapabilities(
            target=self.name,
            specifications=_cuda_capabilities(context, self._status),
            backend_abi_version=self.abi_version,
            extension_abi_version=self._status.abi_version,
        )

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram:
        reduction = compile_reduction(program.operation, context.world_size)
        grouped_reduction = None
        if isinstance(program.algorithm, NativeAllReduce):
            executor_kind = ExecutorKind.NATIVE_ALL_REDUCE
            physical_primitive = PhysicalPrimitive.NCCL_ALL_REDUCE
            stages = (LoweredStage("native_all_reduce", program.wire, True),)
        elif isinstance(program.algorithm, CompressedAllGather):
            executor_kind = ExecutorKind.COMPRESSED_ALL_GATHER
            physical_primitive = PhysicalPrimitive.NCCL_ALL_GATHER_LOCAL_REDUCE
            stages = (
                LoweredStage("quantize_full_contribution", program.wire),
                LoweredStage("compressed_all_gather", program.wire, True),
                LoweredStage(f"fused_dequant_reduce_{reduction.name}", program.wire),
            )
        elif isinstance(program.algorithm, CompressedReduceScatter):
            executor_kind = ExecutorKind.REDUCED_SHARD
            physical_primitive = PhysicalPrimitive.ALL_TO_ALL_LOCAL_REDUCE
            stages = (
                LoweredStage("quantize_destination_chunks", program.wire),
                LoweredStage("quantized_reduce_scatter", program.wire, True),
                LoweredStage(f"fused_dequant_reduce_{reduction.name}", program.wire),
                LoweredStage("return_reduced_shard", program.wire),
            )
        elif isinstance(program.algorithm, CompressedReduceScatterAllGather):
            executor_kind = ExecutorKind.COMPRESSED_RS_AG
            physical_primitive = PhysicalPrimitive.ALL_TO_ALL_QUANTIZED_ALL_GATHER
            stages = (
                LoweredStage("quantize_destination_chunks", program.wire),
                LoweredStage("quantized_reduce_scatter", program.wire, True),
                LoweredStage(
                    f"fused_dequant_reduce_{reduction.name}_requantize",
                    program.wire,
                ),
                LoweredStage("quantized_all_gather", program.wire, True),
                LoweredStage("gathered_dequant_writeback", program.wire),
            )
        elif isinstance(program.algorithm, HierarchicalCompressed):
            if program.algorithm.max_fan_in > 8:
                raise ValueError("hierarchical max_fan_in must be <= 8")
            executor_kind = ExecutorKind.HIERARCHICAL_COMPRESSED
            physical_primitive = PhysicalPrimitive.HIERARCHICAL_COMPRESSED_FULL_TENSOR
            topology = parse_topology_signature(
                context.topology_signature,
                world_size=context.world_size,
            )
            grouped_reduction = compile_grouped_reduction(
                topology,
                max_fan_in=program.algorithm.max_fan_in,
            )
            stages = tuple(
                LoweredStage(
                    f"quantized_group_reduce_level_{index}",
                    program.wire,
                    True,
                )
                for index in range(len(grouped_reduction.levels))
            ) + (
                LoweredStage("normalize_and_requantize_root", program.wire),
            ) + tuple(
                LoweredStage(
                    f"quantized_group_broadcast_level_{index}",
                    program.wire,
                    True,
                )
                for index in reversed(range(len(grouped_reduction.levels)))
            ) + (LoweredStage("gathered_dequant_writeback", program.wire),)
        else:
            raise ValueError("CUDA backend does not yet lower this algorithm")
        stages = _resolve_stages(stages)
        buffer_plan = _compile_buffer_plan(program, context)
        return LoweredProgram(
            self.name,
            program,
            stages,
            reduction,
            context,
            bindings,
            executor_kind,
            physical_primitive,
            buffer_plan,
            grouped_reduction,
        )

    def compile(
        self,
        lowered: LoweredProgram,
    ) -> (
        CudaNativeAllReduceExecutable
        | CudaCompressedAllGatherExecutable
        | CudaReducedShardExecutable
        | CudaFullTensorExecutable
        | CudaHierarchicalFullTensorExecutable
    ):
        if lowered.executor_kind is ExecutorKind.NATIVE_ALL_REDUCE:
            return CudaNativeAllReduceExecutable(lowered)
        if lowered.executor_kind is ExecutorKind.COMPRESSED_ALL_GATHER:
            return CudaCompressedAllGatherExecutable(lowered, self._status)
        if lowered.executor_kind is ExecutorKind.REDUCED_SHARD:
            return CudaReducedShardExecutable(lowered, self._status)
        if lowered.executor_kind is ExecutorKind.COMPRESSED_RS_AG:
            return CudaFullTensorExecutable(lowered, self._status)
        if lowered.executor_kind is ExecutorKind.HIERARCHICAL_COMPRESSED:
            runtime = lowered.bindings.backend_runtime
            if isinstance(runtime, GroupedTransportRuntime):
                assert lowered.grouped_reduction is not None
                bindings = bind_grouped_transport(
                    lowered.grouped_reduction,
                    new_group=runtime.new_group,
                )
                lowered = replace(
                    lowered,
                    bindings=RuntimeBindings(
                        process_group=lowered.bindings.process_group,
                        backend_runtime=bindings,
                        stream_provider=lowered.bindings.stream_provider,
                        allocator=lowered.bindings.allocator,
                    ),
                )
            elif not isinstance(runtime, GroupedTransportBindings):
                raise ValueError(
                    "hierarchical CUDA executable requires GroupedTransportBindings "
                    "or GroupedTransportRuntime"
                )
            return CudaHierarchicalFullTensorExecutable(lowered, self._status)
        raise ValueError(f"CUDA backend cannot compile {lowered.executor_kind.value}")


def _cuda_capabilities(
    context: CompileContext,
    status: CudaExtensionStatus,
) -> tuple[CapabilitySpec, ...]:
    specifications: list[CapabilitySpec] = []
    for operation in ("sum", "mean"):
        specifications.append(
            CapabilitySpec(
                operation=operation,
                output="full_tensor",
                wire="full_precision",
                algorithm="native",
                dtype=context.dtype,
                physical_primitive="nccl_all_reduce",
            )
        )
    if not status.available or (
        status.abi_version is not None and status.abi_version != CudaBackend.abi_version
    ):
        return tuple(specifications)

    if _supports_hierarchical_context(context):
        for operation in ("sum", "mean"):
            specifications.append(
                CapabilitySpec(
                    operation=operation,
                    output="full_tensor",
                    wire="quantized",
                    algorithm="hierarchical_compressed",
                    dtype=context.dtype,
                    bit=8,
                    group_size=64,
                    quant_type="linear",
                    compact=False,
                    physical_primitive="hierarchical_compressed_full_tensor",
                )
            )

    # The production fused reduction kernels currently bind at most eight
    # rank payloads and implement INT8 linear quantization in groups of 64.
    # Keep other codec assets out of capability discovery until a matching
    # executor has passed its own correctness and performance gate.
    if context.world_size > 8:
        return tuple(specifications)

    for operation in ("sum", "mean"):
        for compact in (False, True):
            common = dict(
                operation=operation,
                wire="quantized",
                dtype=context.dtype,
                bit=8,
                group_size=64,
                quant_type="linear",
                compact=compact,
            )
            specifications.extend(
                (
                    CapabilitySpec(
                        **common,
                        output="full_tensor",
                        algorithm="compressed_all_gather",
                        physical_primitive="nccl_all_gather_local_reduce",
                    ),
                    CapabilitySpec(
                        **common,
                        output="reduced_shard",
                        algorithm="compressed_reduce_scatter",
                        physical_primitive="all_to_all_local_reduce",
                    )
                )
            )
        specifications.append(
            CapabilitySpec(
                operation=operation,
                output="full_tensor",
                wire="quantized",
                algorithm="compressed_rs_ag",
                dtype=context.dtype,
                bit=8,
                group_size=64,
                quant_type="linear",
                compact=False,
                physical_primitive="all_to_all_quantized_all_gather",
            )
        )
    return tuple(specifications)


def _resolve_stages(stages: tuple[LoweredStage, ...]) -> tuple[LoweredStage, ...]:
    resolved: list[LoweredStage] = []
    for index, stage in enumerate(stages):
        stage_id = f"stage_{index}"
        dependencies = (resolved[-1].stage_id,) if resolved else ()
        is_output = stage.name.startswith("return_") or stage.name.endswith("writeback")
        kind = (
            StageKind.COLLECTIVE
            if stage.collective
            else StageKind.OUTPUT
            if is_output
            else StageKind.KERNEL
        )
        resolved.append(
            LoweredStage(
                name=stage.name,
                wire=stage.wire,
                collective=stage.collective,
                stage_id=stage_id,
                primitive=stage.name,
                kind=kind,
                dependencies=dependencies,
                stream_role="communication" if stage.collective else "compute",
            )
        )
    return tuple(resolved)


def _compile_buffer_plan(
    program: CommunicationProgram,
    context: CompileContext,
) -> BufferPlan:
    if isinstance(program.algorithm, NativeAllReduce):
        return BufferPlan()
    wire = program.wire
    original_numel = 1
    for size in context.shape:
        original_numel *= size
    element_bytes = {
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.FP32: 4,
    }[context.dtype]
    buffers: list[BufferSpec] = []
    if isinstance(program.algorithm, CompressedAllGather):
        padded_numel = _align(original_numel, wire.group_size)
        payload_numel = _payload_nbytes(original_numel, context.dtype, wire)
        payload_stride = _align(payload_numel, 16)
        buffers.extend(
            (
                BufferSpec(
                    WorkspaceRole.PADDED_INPUT,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
                BufferSpec(
                    WorkspaceRole.SEND,
                    (payload_stride,),
                    "uint8",
                    payload_stride,
                ),
                BufferSpec(
                    WorkspaceRole.RECEIVE,
                    (context.world_size * payload_stride,),
                    "uint8",
                    context.world_size * payload_stride,
                ),
                BufferSpec(
                    WorkspaceRole.LOCAL_RECONSTRUCTION,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
                BufferSpec(
                    WorkspaceRole.RESTORED_SCRATCH,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
            )
        )
        return BufferPlan(tuple(buffers))

    if isinstance(program.algorithm, HierarchicalCompressed):
        padded_numel = _align(original_numel, wire.group_size)
        payload_numel = _payload_nbytes(original_numel, context.dtype, wire)
        payload_stride = _align(payload_numel, 16)
        buffers.extend(
            (
                BufferSpec(
                    WorkspaceRole.PADDED_INPUT,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
                BufferSpec(
                    WorkspaceRole.SEND,
                    (payload_stride,),
                    "uint8",
                    payload_stride,
                ),
                BufferSpec(
                    WorkspaceRole.RECEIVE,
                    (program.algorithm.max_fan_in, payload_stride),
                    "uint8",
                    program.algorithm.max_fan_in * payload_stride,
                ),
                BufferSpec(
                    WorkspaceRole.RESTORED_SCRATCH,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
            )
        )
        return BufferPlan(tuple(buffers))

    shard_numel = _align(
        (original_numel + context.world_size - 1) // context.world_size,
        wire.group_size,
    )
    padded_numel = shard_numel * context.world_size
    payload_numel = _payload_nbytes(shard_numel, context.dtype, wire)
    payload_stride = _align(payload_numel, 16)
    buffers.extend(
        (
            BufferSpec(
                WorkspaceRole.PADDED_INPUT,
                (padded_numel,),
                context.dtype.value,
                padded_numel * element_bytes,
            ),
            BufferSpec(
                WorkspaceRole.SEND,
                (context.world_size, payload_stride),
                "uint8",
                context.world_size * payload_stride,
            ),
            BufferSpec(
                WorkspaceRole.RECEIVE,
                (context.world_size, payload_stride),
                "uint8",
                context.world_size * payload_stride,
            ),
        )
    )
    if isinstance(program.algorithm, CompressedReduceScatterAllGather):
        buffers.extend(
            (
                BufferSpec(
                    WorkspaceRole.LOCAL_RECONSTRUCTION,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
                BufferSpec(
                    WorkspaceRole.RESTORED_SCRATCH,
                    (padded_numel,),
                    context.dtype.value,
                    padded_numel * element_bytes,
                ),
                BufferSpec(
                    WorkspaceRole.REDUCED_PAYLOAD,
                    (payload_stride,),
                    "uint8",
                    payload_stride,
                ),
                BufferSpec(
                    WorkspaceRole.GATHERED_PAYLOAD,
                    (context.world_size * payload_stride,),
                    "uint8",
                    context.world_size * payload_stride,
                ),
            )
        )
    return BufferPlan(tuple(buffers))


def _payload_nbytes(dtype_numel: int, dtype: DataType, wire: object) -> int:
    group_size = wire.group_size
    groups = _align(dtype_numel, group_size) // group_size
    value_bytes = group_size * wire.bit // 8
    scale_bytes = 4 if dtype is DataType.FP32 else 2
    return groups * (value_bytes + scale_bytes)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment if value else 0


def _supports_hierarchical_context(context: CompileContext) -> bool:
    if context.node_count <= 1 or context.topology_signature == "unknown":
        return False
    try:
        topology = parse_topology_signature(
            context.topology_signature,
            world_size=context.world_size,
        )
    except (TypeError, ValueError):
        return False
    return len(topology.node_groups) == context.node_count and all(
        len(group) <= 8 for group in topology.node_groups
    )
