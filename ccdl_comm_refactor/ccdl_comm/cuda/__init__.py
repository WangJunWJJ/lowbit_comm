"""CUDA backend and extension loading helpers for CCDL."""

from .loader import CudaExtensionStatus, load_cuda_extension

__all__ = [
    "CudaAllReduceExecutor",
    "CudaCommunicationBackend",
    "CudaExtensionStatus",
    "CudaReducedShardExecutor",
    "load_cuda_extension",
    "register_cuda_backends",
]


def __getattr__(name: str):
    if name in {"CudaCommunicationBackend", "register_cuda_backends"}:
        from .backend import CudaCommunicationBackend, register_cuda_backends

        return {
            "CudaCommunicationBackend": CudaCommunicationBackend,
            "register_cuda_backends": register_cuda_backends,
        }[name]
    if name in {"CudaAllReduceExecutor", "CudaReducedShardExecutor"}:
        from .executors import CudaAllReduceExecutor, CudaReducedShardExecutor

        return {
            "CudaAllReduceExecutor": CudaAllReduceExecutor,
            "CudaReducedShardExecutor": CudaReducedShardExecutor,
        }[name]
    raise AttributeError(name)
