"""CUDA Backend Protocol implementation."""

from __future__ import annotations

from lowbit_comm.core import (
    BackendCapabilities,
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
        del context
        return BackendCapabilities(
            target=self.name,
            supported_bits=frozenset({4, 8}),
            supported_algorithms=frozenset(
                {"native", "compressed_all_gather", "compressed_reduce_scatter", "compressed_rs_ag"}
            ),
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
