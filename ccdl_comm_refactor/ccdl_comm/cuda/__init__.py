"""CUDA backend and extension loading helpers for CCDL."""

from .loader import CudaExtensionStatus, load_cuda_extension

__all__ = [
    "CudaAllReduceExecutor",
    "CompressedReduceScatterExecutor",
    "CudaCommunicationBackend",
    "CudaExtensionStatus",
    "CudaDynamicGatherExecutor",
    "CudaP2PExecutor",
    "DynamicGatherExecutorCache",
    "CudaReducedShardExecutor",
    "CudaOutputLease",
    "CudaWorkspacePool",
    "WorkspaceKey",
    "WorkspaceLease",
    "WorkspaceStats",
    "load_cuda_extension",
    "compile_dynamic_all_gather",
    "compile_p2p_executor",
    "register_cuda_backends",
]


def __getattr__(name: str):
    if name in {
        "CudaDynamicGatherExecutor",
        "DynamicGatherExecutorCache",
        "compile_dynamic_all_gather",
    }:
        from .dynamic_gather_executor import (
            CudaDynamicGatherExecutor,
            DynamicGatherExecutorCache,
            compile_dynamic_all_gather,
        )

        return {
            "CudaDynamicGatherExecutor": CudaDynamicGatherExecutor,
            "DynamicGatherExecutorCache": DynamicGatherExecutorCache,
            "compile_dynamic_all_gather": compile_dynamic_all_gather,
        }[name]
    if name in {"CudaP2PExecutor", "compile_p2p_executor"}:
        from .p2p_executor import CudaP2PExecutor, compile_p2p_executor

        return {
            "CudaP2PExecutor": CudaP2PExecutor,
            "compile_p2p_executor": compile_p2p_executor,
        }[name]
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
    if name in {"CudaOutputLease", "CudaWorkspacePool", "WorkspaceKey", "WorkspaceLease", "WorkspaceStats"}:
        from .workspace import CudaOutputLease, CudaWorkspacePool, WorkspaceKey, WorkspaceLease, WorkspaceStats

        return {
            "CudaOutputLease": CudaOutputLease,
            "CudaWorkspacePool": CudaWorkspacePool,
            "WorkspaceKey": WorkspaceKey,
            "WorkspaceLease": WorkspaceLease,
            "WorkspaceStats": WorkspaceStats,
        }[name]
    raise AttributeError(name)
