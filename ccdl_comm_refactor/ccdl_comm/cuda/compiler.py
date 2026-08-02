"""Bind existing CUDA codecs and transports into reusable executors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import reduce
from operator import mul
from ccdl_comm.collectives.all_reduce import _run_compressed_all_reduce
from ccdl_comm.collectives.hierarchical import compressed_hierarchical_all_reduce
from ccdl_comm.communication.hierarchical_transport import make_torch_hierarchical_all_reduce
from ccdl_comm.communication.reduce_scatter_transport import make_torch_compressed_reduce_scatter_shard
from ccdl_comm.communication.cuda_completion import (
    CudaCompletionManager,
    native_work_available,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.transports.compressed_reduce_scatter import compile_chunk_plan
from ccdl_comm.execution_info import ExecutionInfo
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.quantization.codec import (
    dequantize_reduce_tensors,
    inplace_dequantize_reduce_mean,
    inplace_dequantize_reduce_mean_update_error_feedback,
    update_error_feedback_residual,
)
from ccdl_comm.quantization.sizing import estimate_quantized_size

from .executors import (
    CudaAllReduceExecutor,
    CudaReducedShardExecutor,
)
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
    execution_info = _execution_info(plan, context, extension_status)
    if plan.collective == "reduce_scatter" and plan.output_layout == "shard":
        return CudaReducedShardExecutor(operation, execution_info)
    precollected_operation = None
    if key == ("all_reduce", "all_gather", "full"):
        precollected_operation = _make_precollected_payload_operation(
            plan,
            context,
            extension_status,
        )
    return CudaAllReduceExecutor(
        operation,
        execution_info,
        precollected_operation=precollected_operation,
    )


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
    completion_manager = CudaCompletionManager(extension_status=extension_status)

    def operation(tensor: object) -> object:
        return _run_compressed_all_reduce(
            tensor,
            config=config,
            op="mean",
            strategy="all_gather",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
            completion_manager=completion_manager,
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
    completion_manager = CudaCompletionManager(extension_status=extension_status)

    def operation(tensor: object) -> object:
        return _run_compressed_all_reduce(
            tensor,
            config=config,
            op="mean",
            strategy="topology",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
            completion_manager=completion_manager,
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
    completion_manager = CudaCompletionManager(extension_status=extension_status)
    chunk_plan = compile_chunk_plan(
        original_numel=reduce(mul, context.shape, 1),
        world_size=context.world_size,
    )
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
    transport = make_torch_compressed_reduce_scatter_shard(
        workspace_cache=workspace_cache,
        completion_manager=completion_manager,
        chunk_plan=chunk_plan,
        fused_dequantize_reduce=(
            inplace_dequantize_reduce_mean if _fused_dequant_fallback_reason(config) is None else None
        ),
        fused_dequantize_reduce_reason=_fused_dequant_fallback_reason(config),
    )

    def operation(tensor: object, *, out: object | None = None) -> object:
        return transport(
            tensor,
            config=config,
            op="mean",
            async_op=plan.async_op,
            dtype=dtype,
            extension_status=extension_status,
            out=out,
        )

    operation.workspace_pool = None if workspace_cache is None else workspace_cache.pool
    operation.chunk_plan = chunk_plan

    return operation


def _require_compression(plan: CommunicationPlan):
    if plan.compression is None:
        raise ValueError("CUDA compressed executor requires compression")
    return plan.compression


def _execution_info(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> ExecutionInfo:
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
    has_native_work = native_work_available(extension_status)
    fused_reduced_shard_reason = _fused_dequant_fallback_reason(config)
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
        fast_path=(
            fast_paths[(plan.collective, plan.strategy, plan.output_layout)]
            if has_native_work
            else "python_fallback"
        ),
        details={
            "device": context.device,
            "dtype": dtype,
            "world_size": context.world_size,
            "topology_signature": context.topology_signature,
            "cuda_fused_reduced_shard": (
                plan.collective == "reduce_scatter"
                and plan.output_layout == "shard"
                and fused_reduced_shard_reason is None
            ),
            "cuda_fused_reduced_shard_fallback_reason": fused_reduced_shard_reason,
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


def _make_precollected_payload_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Callable[..., str | None]:
    config = _require_compression(plan)
    dtype = _normalize_dtype(context.dtype)
    static_fallback_reason = _fused_dequant_fallback_reason(config)
    expected_payload_bytes = estimate_quantized_size(
        reduce(mul, context.shape, 1),
        dtype=dtype,
        config=config,
    ).quantized_bytes

    def run_precollected(
        payloads: object,
        *,
        prepared: object,
        output: object,
        residual: object,
    ) -> str | None:
        buffers = [_payload_buffer(payload) for payload in payloads]
        if not buffers:
            raise ValueError("payloads must not be empty")
        _validate_precollected_payloads(
            buffers,
            expected_bytes=expected_payload_bytes,
            output=output,
        )
        runtime_fallback_reason = None
        if len(buffers) > 8:
            runtime_fallback_reason = f"fused dequant supports at most 8 payloads; received {len(buffers)}"
        used_fused = False
        if static_fallback_reason is None and runtime_fallback_reason is None:
            used_fused = inplace_dequantize_reduce_mean_update_error_feedback(
                buffers,
                prepared,
                output,
                residual,
                config,
                extension_status=extension_status,
                reduce="mean",
            )
        if used_fused:
            return None

        restored = dequantize_reduce_tensors(
            buffers,
            context.shape,
            config,
            dtype=dtype,
            extension_status=extension_status,
            output=output,
            reduce="sum",
        )
        if len(buffers) != 1:
            restored.div_(len(buffers))
        update_error_feedback_residual(
            prepared,
            restored,
            residual,
            extension_status=extension_status,
        )
        reason = (
            static_fallback_reason
            or runtime_fallback_reason
            or "native fused dequant-reduce rejected runtime tensor layout"
        )
        return reason

    return run_precollected


def _fused_dequant_fallback_reason(config: CompressionConfig) -> str | None:
    if config.group_size != 64:
        return f"fused dequant requires group_size=64; received {config.group_size}"
    if config.topk != 0:
        return f"fused dequant requires topk=0; received {config.topk}"
    if config.bit != 8:
        return f"fused dequant requires bit=8; received {config.bit}"
    if config.quant_type != "linear":
        return f"fused dequant requires quant_type='linear'; received {config.quant_type!r}"
    return None


def _payload_buffer(payload: object) -> object:
    return payload.buffer if hasattr(payload, "buffer") else payload


def _validate_precollected_payloads(
    buffers: list[object],
    *,
    expected_bytes: int,
    output: object,
) -> None:
    output_device = getattr(output, "device", None)
    for index, buffer in enumerate(buffers):
        numel = getattr(buffer, "numel", None)
        dtype = getattr(buffer, "dtype", None)
        device = getattr(buffer, "device", None)
        if not callable(numel) or dtype is None or device is None:
            continue
        actual_bytes = int(numel())
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"payload[{index}] must contain {expected_bytes} bytes; "
                f"received {actual_bytes}"
            )
        if str(dtype).removeprefix("torch.") != "uint8":
            raise TypeError(f"payload[{index}] must have dtype uint8; received {dtype}")
        if output_device is not None and device != output_device:
            raise ValueError(
                f"payload[{index}] must be on output device {output_device}; "
                f"received {device}"
            )
