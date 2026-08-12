"""CUDA Backend Protocol implementation."""

from __future__ import annotations

from lowbit_comm.core import (
    BackendCapabilities,
    CommunicationProgram,
    CompileContext,
    CompressedReduceScatter,
    RuntimeBindings,
)
from lowbit_comm.core.lowered import LoweredProgram, LoweredStage

from .executors import CudaReducedShardExecutable
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
            supported_algorithms=frozenset({"compressed_reduce_scatter"}),
        )

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram:
        if not isinstance(program.algorithm, CompressedReduceScatter):
            raise ValueError("CUDA backend does not yet lower this algorithm")
        stages = tuple(
            LoweredStage(name, program.wire)
            for name in (
                "quantize_destination_chunks",
                "quantized_reduce_scatter",
                "fused_dequant_reduce_mean",
                "return_reduced_shard",
            )
        )
        return LoweredProgram(self.name, program, stages, context, bindings)

    def compile(self, lowered: LoweredProgram) -> CudaReducedShardExecutable:
        return CudaReducedShardExecutable(lowered, self._status)
