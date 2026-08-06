"""Bind existing CUDA codecs and transports into reusable executors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul

from ccdl_comm.collectives.all_reduce import _run_compressed_all_reduce
from ccdl_comm.communication.hierarchical_transport import make_group_bound_importer
from ccdl_comm.communication.reduce_scatter_transport import make_torch_compressed_reduce_scatter_shard
from ccdl_comm.communication.cuda_completion import (
    CudaCompletionManager,
    native_work_available,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.transports import (
    PipelinedRingExecutor,
    TreeExecutor,
    compile_chunk_plan,
    compile_pipelined_ring_schedule,
    compile_tree_schedule,
)
from ccdl_comm.cuda.transports.hierarchical import (
    StageExecution,
    compile_hierarchical_stages,
)
from ccdl_comm.cuda.transports.torch_topology import (
    TorchPipelinedRingRuntime,
    TorchTreeRuntime,
)
from ccdl_comm.execution_info import ExecutionInfo
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.quantization.codec import (
    dequantize_tensor,
    dequantize_reduce_tensors,
    inplace_dequantize_reduce_mean,
    inplace_dequantize_reduce_update_local_feedback,
    update_error_feedback_residual,
)
from ccdl_comm.quantization.sizing import estimate_quantized_size
from ccdl_comm.work import CompletionWork

from .executors import (
    CudaAllReduceExecutor,
    CudaReducedShardExecutor,
)
from .loader import CudaExtensionStatus
from .native_collectives import (
    CudaNativeCollectiveExecutor,
    compile_native_collective,
)
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
) -> CudaAllReduceExecutor | CudaNativeCollectiveExecutor | CudaReducedShardExecutor:
    """Compile one validated CUDA plan into a reusable executor."""

    if plan.strategy == "native_nccl":
        return compile_native_collective(
            plan,
            context,
            dist=import_module("torch.distributed"),
        )
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

    factories = {
        ("all_reduce", "all_gather", "full"): _all_gather_operation,
        ("all_reduce", "topology", "full"): _topology_operation,
        ("all_reduce", "hierarchical", "full"): _hierarchical_operation,
        ("reduce_scatter", "compressed", "shard"): _reduced_shard_operation,
    }
    factories.update(
        {
            (collective, "native_nccl", "full"): _native_nccl_operation
            for collective in (
                "all_reduce",
                "all_gather",
                "reduce_scatter",
                "all_to_all",
                "broadcast",
                "reduce",
                "gather",
                "scatter",
                "barrier",
            )
        }
    )
    return factories


def _native_nccl_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    del extension_status
    dist = import_module("torch.distributed")

    def operation(tensor: object) -> object:
        handle = dist.all_reduce(
            tensor,
            op=dist.ReduceOp.AVG,
            group=context.process_group,
            async_op=plan.async_op,
        )
        if not plan.async_op:
            return tensor
        return CompletionWork(
            tensor,
            handle=handle,
            resources=(tensor,),
        )

    return operation


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
    original_numel = reduce(mul, context.shape, 1)
    chunk_plan = compile_chunk_plan(
        original_numel=original_numel,
        world_size=context.world_size,
    )
    ring_aligned = (
        original_numel % context.world_size == 0
        and chunk_plan.shard_numel % config.group_size == 0
    )
    use_ring = context.world_size >= 2 and ring_aligned
    if not use_ring and original_numel % config.group_size != 0:
        raise UnsupportedCollective(
            "all_reduce:topology",
            reason=(
                "native inplace topology dequantization requires a group-aligned "
                f"full tensor or ring shard; numel={original_numel}, "
                f"shard_numel={chunk_plan.shard_numel}, group_size={config.group_size}"
            ),
        )
    topology_method = "ring" if use_ring else "tree"
    workspace_pool = create_torch_workspace_pool(
        max_entries=(plan.workspace_policy.max_entries if plan.workspace_policy.cache else None),
        max_cached_bytes=(
            _workspace_budget(plan, context) if plan.workspace_policy.cache else 0
        ),
    )
    workspace_provider = CudaShardWorkspaceProvider(
        workspace_pool,
        backend=plan.backend,
        collective=plan.collective,
        strategy=f"topology_{topology_method}",
        device=context.device,
        pool_reduced_output=False,
    )
    runtime_kwargs = {
        "config": config,
        "dtype": dtype,
        "world_size": context.world_size,
        "rank": context.rank,
        "participants": _topology_participants(context),
        "extension_status": extension_status,
        "completion_manager": completion_manager,
        "process_group": context.process_group,
    }
    if use_ring:
        runtime = TorchPipelinedRingRuntime(**runtime_kwargs)
        topology_executor = PipelinedRingExecutor(
            schedule=compile_pipelined_ring_schedule(
                chunk_plan=chunk_plan,
                rank=context.rank,
            ),
            runtime=runtime,
            workspace_session_factory=lambda _tensor: workspace_provider.begin(
                stream=runtime.stream
            ),
            completion_manager=completion_manager,
        )
    else:
        runtime = TorchTreeRuntime(**runtime_kwargs)
        topology_executor = TreeExecutor(
            schedule=compile_tree_schedule(
                chunk_plan=chunk_plan,
                rank=context.rank,
            ),
            runtime=runtime,
            workspace_session_factory=lambda _tensor: workspace_provider.begin(
                stream=runtime.stream
            ),
            completion_manager=completion_manager,
        )

    def operation(tensor: object) -> object:
        work = topology_executor.run(tensor)
        return work if plan.async_op else work.wait()

    operation.workspace_pool = workspace_pool
    operation.topology_executor = topology_executor
    operation.topology_method = topology_method
    operation.chunk_plan = chunk_plan
    return operation


def _topology_participants(context: CompileContext) -> tuple[int, ...]:
    if context.process_group is None:
        return tuple(range(context.world_size))
    dist = import_module("torch.distributed")
    getter = getattr(dist, "get_process_group_ranks", None)
    if not callable(getter):
        raise UnsupportedCollective(
            "all_reduce:topology",
            reason="explicit topology process groups require rank introspection",
        )
    participants = tuple(int(rank) for rank in getter(context.process_group))
    if len(participants) != context.world_size:
        raise UnsupportedCollective(
            "all_reduce:topology",
            reason="process group member count does not match compile context world size",
        )
    return participants


def _hierarchical_operation(
    plan: CommunicationPlan,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
) -> Operation:
    dtype = _normalize_dtype(context.dtype)
    completion_manager = CudaCompletionManager(extension_status=extension_status)
    original_shape = tuple(context.shape)
    original_numel = reduce(mul, original_shape, 1)
    streams: dict[str, object] = {}
    workspace_pool = None
    acquire_output = None
    output_owner_token = None
    if plan.workspace_policy.cache:
        config = _require_compression(plan)
        local_world_size = int(context.local_world_size or context.world_size)
        restored_numel = (
            (original_numel + local_world_size - 1) // local_world_size
        ) * local_world_size
        workspace_pool = create_torch_workspace_pool(
            max_entries=plan.workspace_policy.max_entries,
            max_cached_bytes=_workspace_budget(plan, context),
        )
        output_key = WorkspaceKey(
            backend=plan.backend,
            collective=plan.collective,
            strategy=plan.strategy,
            shape_class=(restored_numel,),
            dtype=dtype,
            world_size=context.world_size,
            bit=config.bit,
            group_size=config.group_size,
            chunk_config=(original_numel, restored_numel, local_world_size),
            workspace_kind="full_output",
            device=context.device,
        )
        output_owner_token = object()
        workspace_budget = _workspace_budget(plan, context)

        def acquire_output() -> CudaOutputLease:
            if workspace_budget is not None and output_key.estimated_bytes > workspace_budget:
                raise RuntimeError(
                    "full-output cache budget cannot represent one gather output "
                    f"({output_key.estimated_bytes} bytes > {workspace_budget} bytes)"
                )
            stream = _current_cuda_stream(context.device)
            return CudaOutputLease(
                workspace_pool.acquire(output_key, stream),
                owner_token=output_owner_token,
                completion_manager=completion_manager,
                acquisition_stream=stream,
            )

    def operation_factory(stage, stage_context):
        stream = _new_cuda_stream(stage_context.device)
        streams[stage.name] = stream
        stage_operation = _hierarchical_stage_operation(
            stage,
            stage_context,
            extension_status,
            dtype=dtype,
            original_shape=original_shape,
            original_numel=original_numel,
            completion_manager=completion_manager,
        )
        return _on_cuda_stream(stage_operation, stream, stage_context.device)

    stage_executor = compile_hierarchical_stages(
        plan,
        context,
        operation_factory=operation_factory,
        group_members=lambda group: tuple(
            int(rank)
            for rank in import_module("torch.distributed").get_process_group_ranks(group)
        ),
        stream_factory=lambda stage, stage_context: streams[stage.name],
        completion_factory=lambda value, stream: completion_manager.record_for(
            value,
            stream=stream,
        ),
    )

    def operation(tensor: object, *, out: object | None = None) -> object:
        return stage_executor.run(tensor, out=out)

    operation.hierarchical_executor = stage_executor
    operation.stage_streams = tuple(streams.values())
    operation.workspace_pool = workspace_pool
    operation.output_owner_token = output_owner_token
    operation.acquire_output = acquire_output
    return operation


def _hierarchical_stage_operation(
    stage: object,
    context: CompileContext,
    extension_status: CudaExtensionStatus,
    *,
    dtype: str,
    original_shape: tuple[int, ...],
    original_numel: int,
    completion_manager: CudaCompletionManager,
) -> Operation:
    key = (stage.collective, stage.strategy, stage.output_layout)
    if key == ("reduce_scatter", "compressed", "shard"):
        config = stage.compression
        if not isinstance(config, CompressionConfig):
            raise UnsupportedCollective(
                f"hierarchical:{stage.name}",
                reason="compressed reduce-scatter stage requires compression",
            )
        chunk_plan = compile_chunk_plan(
            original_numel=reduce(mul, context.shape, 1),
            world_size=context.world_size,
        )
        transport = make_torch_compressed_reduce_scatter_shard(
            import_module=make_group_bound_importer(
                context.process_group,
                import_module=import_module,
            ),
            completion_manager=completion_manager,
            chunk_plan=chunk_plan,
        )

        def reduce_scatter(tensor: object) -> StageExecution:
            reduced = transport(
                tensor,
                config=config,
                op="mean",
                async_op=False,
                dtype=dtype,
                extension_status=extension_status,
            )
            return StageExecution(
                reduced.shard,
                resources=(reduced,),
            )

        return reduce_scatter
    if key == ("all_reduce", "topology", "shard"):
        if context.world_size == 1:
            return lambda tensor: tensor
        config = stage.compression
        if not isinstance(config, CompressionConfig):
            raise UnsupportedCollective(
                f"hierarchical:{stage.name}",
                reason="compressed inter-node topology stage requires compression",
            )
        inter_plan = CommunicationPlan(
            "all_reduce",
            "topology",
            backend=stage.backend,
            compression=config,
            output_layout="full",
            async_op=False,
        )
        return _topology_operation(inter_plan, context, extension_status)
    if key == ("all_gather", "native_nccl", "full"):
        bound_import = make_group_bound_importer(
            context.process_group,
            import_module=import_module,
        )
        dist = bound_import("torch.distributed")
        torch = bound_import("torch")

        def restore_full(shard: object, *, out: object | None = None) -> object:
            flat = shard.reshape((-1,))
            restored_numel = int(flat.numel()) * context.world_size
            restored = flat.new_empty((restored_numel,)) if out is None else out
            _validate_full_output_workspace(
                restored,
                flat,
                required_numel=restored_numel,
            )
            gather_into_tensor = getattr(dist, "all_gather_into_tensor", None)
            if callable(gather_into_tensor):
                gather_into_tensor(restored, flat)
            else:
                shards = [flat.new_empty(tuple(flat.shape)) for _ in range(context.world_size)]
                dist.all_gather(shards, flat)
                gathered = torch.cat(shards, dim=0)
                if out is None:
                    restored = gathered
                else:
                    restored.copy_(gathered)
            if restored_numel == original_numel and tuple(restored.shape) == original_shape:
                return restored
            return restored[:original_numel].reshape(original_shape)

        return restore_full
    raise UnsupportedCollective(
        f"hierarchical:{stage.name}",
        reason=(
            "CUDA hierarchical executor does not implement stage "
            f"{stage.collective}:{stage.strategy}:{stage.output_layout}"
        ),
    )


def _new_cuda_stream(device: str) -> object:
    torch = import_module("torch")
    return torch.cuda.Stream(device=device)


def _validate_full_output_workspace(
    output: object,
    shard: object,
    *,
    required_numel: int,
) -> None:
    if int(output.numel()) != required_numel:
        raise ValueError(
            "full output workspace must contain exactly "
            f"{required_numel} elements, got {int(output.numel())}"
        )
    if getattr(output, "dtype", None) != getattr(shard, "dtype", None):
        raise ValueError("full output workspace dtype must match the reduced shard")
    if getattr(output, "device", None) != getattr(shard, "device", None):
        raise ValueError("full output workspace device must match the reduced shard")
    is_contiguous = getattr(output, "is_contiguous", None)
    if callable(is_contiguous) and not bool(is_contiguous()):
        raise ValueError("full output workspace must be contiguous")


def _on_cuda_stream(operation: Operation, stream: object, device: str) -> Operation:
    torch = import_module("torch")

    def launch(value: object, *, out: object | None = None) -> object:
        current_stream = torch.cuda.current_stream(device=device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            return operation(value) if out is None else operation(value, out=out)

    return launch


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
    dtype = _normalize_dtype(context.dtype)
    numel = reduce(mul, context.shape, 1)
    config = plan.compression
    native_nccl = plan.strategy == "native_nccl"
    if native_nccl:
        original_bytes = numel * _dtype_bytes(dtype)
        compressed_bytes = original_bytes
        compression_ratio = 1.0
    else:
        config = _require_compression(plan)
        estimate = estimate_quantized_size(numel, dtype=dtype, config=config)
        original_bytes = estimate.original_bytes
        compressed_bytes = estimate.quantized_bytes
        compression_ratio = (
            estimate.compression_ratio if estimate.quantized_bytes else 1.0
        )
    fast_paths = {
        ("all_reduce", "native_nccl", "full"): "cuda_native_nccl",
        ("all_reduce", "all_gather", "full"): "cuda_all_gather",
        ("all_reduce", "topology", "full"): "cuda_topology",
        ("all_reduce", "hierarchical", "full"): "cuda_hierarchical",
        ("reduce_scatter", "compressed", "shard"): "cuda_reduced_shard",
    }
    has_native_work = native_nccl or native_work_available(extension_status)
    if plan.collective == "reduce_scatter" and plan.output_layout == "shard":
        fused_capability = fused_reduced_shard_capability or _fused_reduced_shard_capability(
            config,
            dtype=dtype,
            world_size=context.world_size,
            extension_status=extension_status,
        )
    else:
        fused_capability = None
    hierarchical_reason = None
    if plan.strategy == "hierarchical":
        hierarchical_reason = (
            "single-node hierarchical is explicit-only because the validated auto "
            "topology path remains faster"
            if context.node_count == 1
            else "multi-node hierarchical has not passed a production performance gate"
        )
    return ExecutionInfo(
        requested_strategy=plan.strategy,
        executed_strategy=plan.strategy,
        backend="cuda",
        fallback_used=False,
        fallback_reason=None,
        stage_names=tuple(stage.name for stage in plan.stages),
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=compression_ratio,
        workspace_cache_hit=False,
        async_capable=plan.strategy != "hierarchical",
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
            "hierarchical_recommended": (
                False if plan.strategy == "hierarchical" else None
            ),
            "hierarchical_recommendation_reason": hierarchical_reason,
        },
    )


def _dtype_bytes(dtype: str) -> int:
    return {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype, 4)


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
    local_input_index = _process_group_rank(context)

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
        if local_input_index < 0 or local_input_index >= len(buffers):
            raise RuntimeError(
                "compiled process-group rank is outside the gathered payload order: "
                f"rank={local_input_index}, payloads={len(buffers)}"
            )
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
            used_fused = inplace_dequantize_reduce_update_local_feedback(
                buffers,
                local_input_index,
                prepared,
                output,
                residual,
                config,
                extension_status=extension_status,
                reduce=plan.reduce_op,
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
        if plan.reduce_op == "mean" and len(buffers) != 1:
            restored.div_(len(buffers))
        local_restored = dequantize_tensor(
            buffers[local_input_index],
            context.shape,
            config,
            dtype=dtype,
            extension_status=extension_status,
        )
        update_error_feedback_residual(
            prepared,
            local_restored,
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


def _process_group_rank(context: CompileContext) -> int:
    if context.process_group is None:
        return context.rank
    try:
        dist = import_module("torch.distributed")
        is_initialized = getattr(dist, "is_initialized", None)
        if callable(is_initialized) and is_initialized():
            return int(dist.get_rank(context.process_group))
    except (ImportError, ModuleNotFoundError, RuntimeError, TypeError, ValueError):
        pass
    return context.rank


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
