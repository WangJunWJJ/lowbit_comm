"""CUDA Backend Protocol implementation."""

from __future__ import annotations

from lowbit_comm.core import (
    BackendCapabilities,
    CapabilitySpec,
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    RuntimeBindings,
    NativeAllReduce,
    compile_reduction,
)
from lowbit_comm.core.lowered import LoweredProgram, LoweredStage

from .executors import (
    CudaCompressedAllGatherExecutable,
    CudaFullTensorExecutable,
    CudaNativeAllReduceExecutable,
    CudaReducedShardExecutable,
)
from .loader import CudaExtensionStatus, load_cuda_extension


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
        if isinstance(program.algorithm, NativeAllReduce):
            stages = (LoweredStage("native_all_reduce", program.wire, True),)
        elif isinstance(program.algorithm, CompressedAllGather):
            stages = (
                LoweredStage("quantize_full_contribution", program.wire),
                LoweredStage("compressed_all_gather", program.wire, True),
                LoweredStage(f"fused_dequant_reduce_{reduction.name}", program.wire),
            )
        elif isinstance(program.algorithm, CompressedReduceScatter):
            stages = (
                LoweredStage("quantize_destination_chunks", program.wire),
                LoweredStage("quantized_reduce_scatter", program.wire, True),
                LoweredStage(f"fused_dequant_reduce_{reduction.name}", program.wire),
                LoweredStage("return_reduced_shard", program.wire),
            )
        elif isinstance(program.algorithm, CompressedReduceScatterAllGather):
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
        else:
            raise ValueError("CUDA backend does not yet lower this algorithm")
        return LoweredProgram(
            self.name,
            program,
            stages,
            reduction,
            context,
            bindings,
        )

    def compile(
        self,
        lowered: LoweredProgram,
    ) -> (
        CudaNativeAllReduceExecutable
        | CudaCompressedAllGatherExecutable
        | CudaReducedShardExecutable
        | CudaFullTensorExecutable
    ):
        if isinstance(lowered.program.algorithm, NativeAllReduce):
            return CudaNativeAllReduceExecutable(lowered)
        if isinstance(lowered.program.algorithm, CompressedAllGather):
            return CudaCompressedAllGatherExecutable(lowered, self._status)
        if isinstance(lowered.program.algorithm, CompressedReduceScatter):
            return CudaReducedShardExecutable(lowered, self._status)
        if isinstance(
            lowered.program.algorithm,
            CompressedReduceScatterAllGather,
        ):
            return CudaFullTensorExecutable(lowered, self._status)
        raise ValueError("CUDA backend does not yet compile this algorithm")


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

    for operation in ("sum", "mean"):
        for bit in (4, 8):
            for group_size in (16, 32, 64):
                for compact in (False, True):
                    common = dict(
                        operation=operation,
                        wire="quantized",
                        dtype=context.dtype,
                        bit=bit,
                        group_size=group_size,
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
                                physical_primitive="nccl_all_to_all_local_reduce",
                            ),
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
                physical_primitive="nccl_all_to_all_quantized_all_gather",
            )
        )
    return tuple(specifications)
