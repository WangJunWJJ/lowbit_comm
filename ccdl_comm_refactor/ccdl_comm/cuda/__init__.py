"""CUDA backend and extension loading helpers for CCDL."""

from .loader import CudaExtensionStatus, load_cuda_extension

__all__ = [
    "CudaAllReduceExecutor",
    "CompressedReduceScatterExecutor",
    "CudaCommunicationBackend",
    "CudaExtensionStatus",
    "CudaReducedShardExecutor",
    "CudaWorkspacePool",
    "WorkspaceKey",
    "WorkspaceLease",
    "WorkspaceStats",
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
    if name in {"CompressedReduceScatterExecutor", "CudaAllReduceExecutor", "CudaReducedShardExecutor"}:
        from .executors import CompressedReduceScatterExecutor, CudaAllReduceExecutor, CudaReducedShardExecutor

        return {
            "CompressedReduceScatterExecutor": CompressedReduceScatterExecutor,
            "CudaAllReduceExecutor": CudaAllReduceExecutor,
            "CudaReducedShardExecutor": CudaReducedShardExecutor,
        }[name]
    if name in {"CudaWorkspacePool", "WorkspaceKey", "WorkspaceLease", "WorkspaceStats"}:
        from .workspace import CudaWorkspacePool, WorkspaceKey, WorkspaceLease, WorkspaceStats

        return {
            "CudaWorkspacePool": CudaWorkspacePool,
            "WorkspaceKey": WorkspaceKey,
            "WorkspaceLease": WorkspaceLease,
            "WorkspaceStats": WorkspaceStats,
        }[name]
    raise AttributeError(name)
