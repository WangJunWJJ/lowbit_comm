"""CUDA Backend Protocol implementation."""

from __future__ import annotations

from lowbit_comm.core import (
    BackendCapabilities,
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    RuntimeBindings,
)
from lowbit_comm.core.lowered import LoweredProgram, LoweredStage

from .executors import CudaFullTensorExecutable, CudaReducedShardExecutable
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
                {"compressed_reduce_scatter", "compressed_rs_ag"}
            ),
        )

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram:
        if isinstance(program.algorithm, CompressedReduceScatter):
            stages = (
                LoweredStage("quantize_destination_chunks", program.wire),
                LoweredStage("quantized_reduce_scatter", program.wire, True),
                LoweredStage("fused_dequant_reduce_mean", program.wire),
                LoweredStage("return_reduced_shard", program.wire),
            )
        elif isinstance(program.algorithm, CompressedReduceScatterAllGather):
            stages = (
                LoweredStage("quantize_destination_chunks", program.wire),
                LoweredStage("quantized_reduce_scatter", program.wire, True),
                LoweredStage("fused_dequant_reduce_mean_requantize", program.wire),
                LoweredStage("quantized_all_gather", program.wire, True),
                LoweredStage("gathered_dequant_writeback", program.wire),
            )
        else:
            raise ValueError("CUDA backend does not yet lower this algorithm")
        return LoweredProgram(self.name, program, stages, context, bindings)

    def compile(
        self,
        lowered: LoweredProgram,
    ) -> CudaReducedShardExecutable | CudaFullTensorExecutable:
        if isinstance(lowered.program.algorithm, CompressedReduceScatter):
            return CudaReducedShardExecutable(lowered, self._status)
        if isinstance(
            lowered.program.algorithm,
            CompressedReduceScatterAllGather,
        ):
            return CudaFullTensorExecutable(lowered, self._status)
        raise ValueError("CUDA backend does not yet compile this algorithm")
