"""Bind existing CUDA codecs and transports into reusable executors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import reduce
from operator import mul
from typing import Any

from ccdl_comm.collectives.all_reduce import _run_compressed_all_reduce
from ccdl_comm.collectives.hierarchical import compressed_hierarchical_all_reduce
from ccdl_comm.communication.hierarchical_transport import make_torch_hierarchical_all_reduce
from ccdl_comm.communication.reduce_scatter_transport import make_torch_compressed_reduce_scatter_shard
from ccdl_comm.execution_info import ExecutionInfo
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.quantization.sizing import estimate_quantized_size

from .executors import CudaAllReduceExecutor, CudaReducedShardExecutor
from .loader import CudaExtensionStatus
from .workspace import CudaShardWorkspaceProvider, create_torch_workspace_pool


Operation = Callable[[object], object]
OperationFactory = Callable[[CommunicationPlan, CompileContext, CudaExtensionStatus], Operation]
OperationKey = tuple[str, str, str]


def compile_cuda_plan(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
    *,
    operation_factories: Mapping[OperationKey, OperationFactory] | None = None,
) -> CudaAllReduceExecutor | CudaReducedShardExecutor:
    """Compile one validated CUDA plan into a reusable executor."""

    factories = default_operation_factories() if operation_factories is None else operation_factories
    key = (plan.collective, plan.strategy, plan.output_layout)
    operation = factories[key](plan, context, extension_status)
    execution_info = _execution_info(plan, context)
    if plan.collective == "reduce_scatter" and plan.output_layout == "shard":
        return CudaReducedShardExecutor(operation, execution_info)
    return CudaAllReduceExecutor(operation, execution_info)


def default_operation_factories() -> dict[OperationKey, OperationFactory]:
    """Return fresh bindings for the currently validated production paths."""

    return {
        ("all_reduce", "all_gather", "full"): _all_gather_operation,
        ("all_reduce", "topology", "full"): _topology_operation,
        ("reduce_scatter", "compressed", "shard"): _reduced_shard_operation,
    }


def _all_gather_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)

    def operation(tensor: object) -> object:
        return _run_compressed_all_reduce(
            tensor,
            config=config,
            op="mean",
            strategy="all_gather",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
            process_group=context.process_group,
        )

    return operation


def _topology_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)

    def operation(tensor: object) -> object:
        return _run_compressed_all_reduce(
            tensor,
            config=config,
            op="mean",
            strategy="topology",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
            process_group=context.process_group,
        )

    return operation


def _hierarchical_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)
    local_group_size = context.local_world_size or min(context.world_size, 2)
    transport = make_torch_hierarchical_all_reduce(local_group_size=local_group_size)

    def operation(tensor: object) -> object:
        return compressed_hierarchical_all_reduce(
            tensor,
            config=config,
            op="mean",
            async_op=plan.async_op,
            dtype=dtype,
            hierarchical_all_reduce=transport,
            extension_status=extension_status,
        )

    return operation


def _reduced_shard_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)
    workspace_cache = None
    if plan.workspace_policy.cache:
        workspace_pool = create_torch_workspace_pool(
            max_entries=plan.workspace_policy.max_entries,
            max_cached_bytes=_workspace_budget(plan, context),
        )
        workspace_cache = CudaShardWorkspaceProvider(
            workspace_pool,
            backend=plan.backend,
            collective=plan.collective,
            strategy=plan.strategy,
            device=context.device,
            pool_reduced_output=False,
        )
    transport = make_torch_compressed_reduce_scatter_shard(workspace_cache=workspace_cache)

    def operation(tensor: object) -> object:
        return transport(
            tensor,
            config=config,
            op="mean",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
        )

    operation.workspace_pool = None if workspace_cache is None else workspace_cache.pool

    return operation


def _require_compression(plan: CommunicationPlan):
    if plan.compression is None:
        raise ValueError("CUDA compressed executor requires compression")
    return plan.compression


def _execution_info(plan: CommunicationPlan, context: CompileContext) -> ExecutionInfo:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)
    numel = reduce(mul, context.shape, 1)
    estimate = estimate_quantized_size(numel, dtype=dtype, config=config)
    fast_paths = {
        ("all_reduce", "all_gather", "full"): "cuda_all_gather",
        ("all_reduce", "topology", "full"): "cuda_topology",
        ("all_reduce", "hierarchical", "full"): "cuda_hierarchical",
        ("reduce_scatter", "compressed", "shard"): "cuda_reduced_shard",
    }
    return ExecutionInfo(
        requested_strategy=plan.strategy,
        executed_strategy=plan.strategy,
        backend="cuda",
        fallback_used=False,
        fallback_reason=None,
        stage_names=tuple(stage.name for stage in plan.stages),
        original_bytes=estimate.original_bytes,
        compressed_bytes=estimate.quantized_bytes,
        compression_ratio=estimate.compression_ratio if estimate.quantized_bytes else 1.0,
        workspace_cache_hit=False,
        async_capable=plan.strategy not in {"topology", "hierarchical"},
        fast_path=fast_paths[(plan.collective, plan.strategy, plan.output_layout)],
        details={
            "device": context.device,
            "dtype": dtype,
            "world_size": context.world_size,
            "topology_signature": context.topology_signature,
        },
    )


def _normalize_dtype(dtype: str) -> str:
    normalized = dtype.strip().lower().removeprefix("torch.")
    return {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }.get(normalized, normalized)


def _workspace_budget(
    plan: CommunicationPlan,
    context: CompileContext,
) -> int | None:
    limits = tuple(
        limit
        for limit in (
            plan.workspace_policy.max_cached_bytes,
            context.workspace_budget_bytes,
        )
        if limit is not None
    )
    return min(limits) if limits else None
