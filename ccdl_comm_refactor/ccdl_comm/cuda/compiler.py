"""Bind existing CUDA codecs and transports into reusable executors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
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
from .workspace import (
    CudaOutputLease,
    CudaShardWorkspaceProvider,
    WorkspaceKey,
    create_torch_workspace_pool,
)


Operation = Callable[[object], object]
OperationFactory = Callable[[CommunicationPlan, CompileContext, CudaExtensionStatus], Operation]
OperationKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class FusedReducedShardCapability:
    """Compile-time eligibility for the fused ReducedShard output path."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise ValueError("available fused ReducedShard capability must not have a fallback reason")
        if not self.available and not self.reason:
            raise ValueError("unavailable fused ReducedShard capability requires a fallback reason")


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
    execution_info = _execution_info(
        plan,
        context,
        extension_status,
        fused_reduced_shard_capability=getattr(operation, "fused_reduced_shard_capability", None),
    )
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
    fused_capability = _fused_reduced_shard_capability(
        config,
        dtype=dtype,
        world_size=context.world_size,
        extension_status=extension_status,
    )
    workspace_cache = None
    acquire_output = None
    output_owner_token = None
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
        output_key = WorkspaceKey(
            backend=plan.backend,
            collective=plan.collective,
            strategy=plan.strategy,
            shape_class=(chunk_plan.shard_numel,),
            dtype=dtype,
            world_size=context.world_size,
            bit=config.bit,
            group_size=config.group_size,
            chunk_config=(chunk_plan.original_numel, chunk_plan.shard_numel),
            workspace_kind="reduced_output",
            device=context.device,
        )
        output_owner_token = object()
        workspace_budget = _workspace_budget(plan, context)

        def acquire_output() -> CudaOutputLease:
            if workspace_budget is not None and output_key.estimated_bytes > workspace_budget:
                raise RuntimeError(
                    "ReducedShard output cache budget cannot represent one shard output "
                    f"({output_key.estimated_bytes} bytes > {workspace_budget} bytes)"
                )
            stream = _current_cuda_stream(context.device)
            return CudaOutputLease(
                workspace_pool.acquire(output_key, stream),
                owner_token=output_owner_token,
                completion_manager=completion_manager,
                acquisition_stream=stream,
            )
    transport = make_torch_compressed_reduce_scatter_shard(
        workspace_cache=workspace_cache,
        completion_manager=completion_manager,
        chunk_plan=chunk_plan,
        fused_dequantize_reduce=(
            _bind_fused_reduced_shard_callback(config, extension_status)
            if fused_capability.available
            else None
        ),
        fused_dequantize_reduce_reason=fused_capability.reason,
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
    operation.fused_reduced_shard_capability = fused_capability
    operation.output_owner_token = output_owner_token
    operation.acquire_output = acquire_output

    return operation


def _require_compression(plan: CommunicationPlan):
    if plan.compression is None:
        raise ValueError("CUDA compressed executor requires compression")
    return plan.compression


def _execution_info(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
    *,
    fused_reduced_shard_capability: FusedReducedShardCapability | None = None,
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
    if plan.collective == "reduce_scatter" and plan.output_layout == "shard":
        fused_capability = fused_reduced_shard_capability or _fused_reduced_shard_capability(
            config,
            dtype=dtype,
            world_size=context.world_size,
            extension_status=extension_status,
        )
    else:
        fused_capability = None
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
            "cuda_fused_reduced_shard": fused_capability is not None and fused_capability.available,
            "cuda_fused_reduced_shard_fallback_reason": (
                None if fused_capability is None else fused_capability.reason
            ),
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


def _current_cuda_stream(device: str) -> object | None:
    """Return the consumer stream when CUDA is available, otherwise ``None``."""

    try:
        torch = import_module("torch")
    except (ImportError, ModuleNotFoundError):
        return None
    cuda = getattr(torch, "cuda", None)
    current_stream = getattr(cuda, "current_stream", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(current_stream) or not callable(is_available) or not is_available():
        return None
    try:
        return current_stream(device=device)
    except (RuntimeError, TypeError):
        return current_stream()


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


def _fused_reduced_shard_capability(
    config: CompressionConfig,
    *,
    dtype: str,
    world_size: int,
    extension_status: CudaExtensionStatus,
) -> FusedReducedShardCapability:
    reason = _fused_dequant_fallback_reason(config)
    if reason is not None:
        return FusedReducedShardCapability(False, reason)
    if dtype not in {"fp16", "bf16", "fp32"}:
        return FusedReducedShardCapability(
            False,
            f"fused dequant requires dtype fp16, bf16, or fp32; received {dtype!r}",
        )
    if world_size > 8:
        return FusedReducedShardCapability(
            False,
            f"fused dequant supports at most 8 input ranks; received {world_size}",
        )
    if not extension_status.available or extension_status.module is None:
        return FusedReducedShardCapability(
            False,
            extension_status.reason or "CCDL CUDA extension is unavailable",
        )
    if not callable(getattr(extension_status.module, "inplace_dequantize_reduce_mean", None)):
        return FusedReducedShardCapability(
            False,
            "CCDL CUDA extension does not export inplace_dequantize_reduce_mean",
        )
    return FusedReducedShardCapability(True)


def _bind_fused_reduced_shard_callback(
    config: CompressionConfig,
    extension_status: CudaExtensionStatus,
) -> Callable[..., bool]:
    """Adapt the transport's minimal ABI to the Task 1 codec facade once."""

    def fused_dequantize_reduce(buffers: list[object], output: object, *, reduce: str) -> bool:
        return inplace_dequantize_reduce_mean(
            buffers,
            output,
            config,
            extension_status=extension_status,
            reduce=reduce,
        )

    return fused_dequantize_reduce


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
