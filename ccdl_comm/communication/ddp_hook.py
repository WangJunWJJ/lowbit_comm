from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.communication.async_pipeline import AsyncBucketPipeline
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.ddp import DDPBucketProcessor
from ccdl_comm.communication.gather_reduce import CompressedAllGatherReduce, GatheredPayloads
from ccdl_comm.communication.payload_packing import (
    DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    make_fused_payload_all_gather,
    make_payload_all_gather,
    should_fuse_payload,
)
from ccdl_comm.communication.strategy import CollectiveCapabilities, plan_ddp_compression_strategy
from ccdl_comm.communication.torch_transport import (
    make_torch_all_gather,
    make_torch_async_all_gather,
    make_torch_tensor_all_reduce,
)
from ccdl_comm.communication.transport_capability import require_compressed_transport
from ccdl_comm.communication.topology_transport import make_legacy_topology_all_reduce
from ccdl_comm.communication.workspace import DequantizedWorkspaceCache
from ccdl_comm.config import CompressionConfig
from ccdl_comm.collectives.hierarchical import compressed_hierarchical_all_reduce
from ccdl_comm.collectives.reduce_scatter import compressed_reduce_scatter
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.quantization.codec import (
    allocate_dequantized_buffer,
    dequantize_reduce_tensors,
    dequantize_reduce_update_error_feedback,
    dequantize_tensor,
    inplace_dequantize_reduce_mean_update_error_feedback,
    quantize_tensor,
)
from ccdl_comm.quantization.error_feedback import ErrorFeedbackState
from ccdl_comm.quantization.error_feedback_policy import ErrorFeedbackPolicy


def _torch_future_factory() -> Any:
    torch = import_module("torch")
    return torch.futures.Future()


def create_ddp_comm_hook(
    config: CompressionConfig,
    *,
    dtype: str = "auto",
    strategy: str = "native_nccl",
    reduce: str = "mean",
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None,
    all_gather: Callable[[Any], GatheredPayloads] | None = None,
    async_gather: bool = False,
    async_error_feedback: bool = False,
    synchronize_async_feedback_completion: bool = True,
    async_all_gather: Callable[[Any], Any] | None = None,
    native_error_feedback_update: Callable[[Any, Any, Any], Any] | None = None,
    native_dequantize_reduce_update_feedback: Callable[..., Any] | None = None,
    native_inplace_dequantize_reduce_update_feedback: Callable[..., bool] | None = None,
    reduce_scatter_all_gather: Callable[..., Any] | None = None,
    hierarchical_all_reduce: Callable[..., Any] | None = None,
    topology_all_reduce: Callable[..., Any] | None = None,
    topology_method: str | None = None,
    allocate_dequantized_workspace: Callable[[Any, tuple[int, ...], CompressionConfig], Any] | None = None,
    workspace_cache_max_entries: int | None = 1,
    workspace_cache_max_bytes: int | None = None,
    completion_manager: CudaCompletionManager | Any | None = None,
    fuse_payload: bool = False,
    fuse_payload_min_numel: int = DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    min_compress_numel: int = 0,
    bypass_all_reduce: Callable[[Any, str], Any] | None = None,
    error_feedback: ErrorFeedbackState | None = None,
    extension_status: CudaExtensionStatus | None = None,
    future_factory: Callable[[], Any] = _torch_future_factory,
    annotation_provider: Callable[[], dict[str, Any]] | None = None,
) -> Callable[[Any, Any], Any]:
    """Create a PyTorch DDP comm hook backed by CCDL bucket processing."""

    strategy_plan = plan_ddp_compression_strategy(
        requested_strategy=strategy,
        world_size=_distributed_world_size(default=1),
        rank=_distributed_rank(default=0),
        local_world_size=_env_int("LOCAL_WORLD_SIZE"),
        node_count=_env_int("NODE_COUNT"),
        capabilities=CollectiveCapabilities(
            reduce_scatter=reduce_scatter_all_gather is not None,
            hierarchical=hierarchical_all_reduce is not None,
            topology=topology_all_reduce is not None or strategy == "topology",
        ),
    )
    effective_strategy = strategy_plan.strategy
    if strategy_plan.requires_fallback and effective_strategy in {"reduce_scatter", "hierarchical", "topology"}:
        effective_strategy = strategy_plan.fallback_strategy

    def active_quantize(tensor: Any, active_config: CompressionConfig) -> Any:
        if quantize is not None:
            return quantize(tensor, active_config)
        return quantize_tensor(tensor, active_config, extension_status=extension_status)

    def active_dequantize(payload: Any, shape: tuple[int, ...], active_config: CompressionConfig, active_dtype: str) -> Any:
        if dequantize is not None:
            return dequantize(payload, shape, active_config, active_dtype)
        return dequantize_tensor(
            _payload_buffer(payload),
            shape,
            active_config,
            dtype=active_dtype,
            extension_status=extension_status,
        )

    feedback = error_feedback or ErrorFeedbackState()
    feedback_policy = ErrorFeedbackPolicy(config)
    native_all_reduce = bypass_all_reduce or make_torch_tensor_all_reduce()
    active_completion_manager = completion_manager or CudaCompletionManager()
    active_native_dequantize_reduce_update_feedback = (
        native_dequantize_reduce_update_feedback or dequantize_reduce_update_error_feedback
    )
    active_native_inplace_dequantize_reduce_update_feedback = (
        native_inplace_dequantize_reduce_update_feedback or inplace_dequantize_reduce_mean_update_error_feedback
    )
    workspace_cache = DequantizedWorkspaceCache(
        allocator=allocate_dequantized_workspace or allocate_dequantized_buffer,
        max_entries=workspace_cache_max_entries,
        max_cached_bytes=workspace_cache_max_bytes,
    )

    if effective_strategy == "native_nccl":

        def process_bucket(bucket: Any) -> Any:
            return native_all_reduce(_clone_tensor(bucket.buffer()), reduce)

    elif effective_strategy == "reduce_scatter":

        def process_bucket(bucket: Any) -> Any:
            tensor = bucket.buffer()
            if not _should_compress(tensor, min_numel=min_compress_numel):
                return native_all_reduce(_clone_tensor(tensor), reduce)
            return compressed_reduce_scatter(
                tensor,
                config=config,
                op=reduce,
                async_op=False,
                dtype=_resolve_dtype(dtype, tensor),
                reduce_scatter=reduce_scatter_all_gather,
                extension_status=extension_status,
            )

    elif effective_strategy == "hierarchical":

        def process_bucket(bucket: Any) -> Any:
            tensor = bucket.buffer()
            if not _should_compress(tensor, min_numel=min_compress_numel):
                return native_all_reduce(_clone_tensor(tensor), reduce)
            return compressed_hierarchical_all_reduce(
                tensor,
                config=config,
                op=reduce,
                async_op=False,
                dtype=_resolve_dtype(dtype, tensor),
                hierarchical_all_reduce=hierarchical_all_reduce,
                extension_status=extension_status,
            )

    elif effective_strategy == "topology":
        active_topology_all_reduce = topology_all_reduce or make_legacy_topology_all_reduce(method=topology_method)

        def process_bucket(bucket: Any) -> Any:
            tensor = bucket.buffer()
            if not _should_compress(tensor, min_numel=min_compress_numel):
                return native_all_reduce(_clone_tensor(tensor), reduce)
            return active_topology_all_reduce(
                tensor,
                config=config,
                op=reduce,
                async_op=False,
                dtype=_resolve_dtype(dtype, tensor),
                extension_status=extension_status,
            )

    elif effective_strategy == "all_gather":
        if all_gather is not None:
            normal_all_gather = all_gather
            fused_all_gather = all_gather
        else:
            buffer_all_gather = make_torch_all_gather()
            normal_all_gather = make_payload_all_gather(buffer_all_gather)
            fused_all_gather = make_fused_payload_all_gather(buffer_all_gather)
        active_async_all_gather = async_all_gather or make_torch_async_all_gather()

        def process_bucket(bucket: Any) -> Any:
            key = bucket.index() if callable(getattr(bucket, "index", None)) else id(bucket)
            original = bucket.buffer()
            if not _should_compress(original, min_numel=min_compress_numel):
                return native_all_reduce(_clone_tensor(original), reduce)
            feedback_decision = feedback_policy.decide(key, numel=_numel(original))
            prepared = feedback.compensate(key, original) if feedback_decision.apply else original
            active_dtype = _resolve_dtype(dtype, prepared)
            active_all_gather = (
                fused_all_gather
                if should_fuse_payload(prepared, enabled=fuse_payload, min_numel=fuse_payload_min_numel)
                else normal_all_gather
            )
            if dequantize is None and reduce in {"mean", "sum"}:
                local_payload = _coerce_payload(
                    active_quantize(prepared, config),
                    shape=tuple(prepared.shape),
                    dtype=active_dtype,
                )
                needs_feedback = feedback_decision.apply or feedback_decision.update
                use_async_gather = async_gather and (async_error_feedback or not needs_feedback)
                if use_async_gather:
                    gather_work = active_async_all_gather(_payload_buffer(local_payload))
                    outer_future = future_factory()

                    if needs_feedback:
                        get_residual = getattr(feedback, "get", None)
                        residual = get_residual(key) if callable(get_residual) else None
                        combined_updated = [False]

                        def dequantize_reduce_feedback(gathered: GatheredPayloads) -> Any:
                            buffers = [_payload_buffer(payload) for payload in gathered.payloads]
                            if feedback_decision.update and residual is not None:
                                try:
                                    restored_workspace = workspace_cache.get(
                                        key,
                                        prepared,
                                        tuple(prepared.shape),
                                        config,
                                    )
                                    used_inplace = active_native_inplace_dequantize_reduce_update_feedback(
                                        buffers,
                                        prepared,
                                        restored_workspace,
                                        residual,
                                        config,
                                        extension_status=extension_status,
                                        reduce=reduce,
                                    )
                                    if used_inplace:
                                        combined_updated[0] = True
                                        return _reshape_to_shape(restored_workspace, tuple(prepared.shape))
                                except Exception:
                                    pass
                                try:
                                    restored = active_native_dequantize_reduce_update_feedback(
                                        buffers,
                                        prepared,
                                        residual,
                                        tuple(prepared.shape),
                                        config,
                                        dtype=active_dtype,
                                        extension_status=extension_status,
                                        reduce=reduce,
                                    )
                                    combined_updated[0] = True
                                    return restored
                                except Exception:
                                    pass
                            return dequantize_reduce_tensors(
                                buffers,
                                tuple(prepared.shape),
                                config,
                                dtype=active_dtype,
                                extension_status=extension_status,
                                reduce=reduce,
                            )

                        def update_feedback(restored: Any) -> None:
                            if not feedback_decision.update:
                                return
                            if combined_updated[0]:
                                return
                            latest_residual = get_residual(key) if callable(get_residual) else None
                            if native_error_feedback_update is not None and latest_residual is not None:
                                native_error_feedback_update(prepared, restored, latest_residual)
                                return
                            feedback.update(key, original=prepared, transmitted=restored)

                        return AsyncBucketPipeline(
                            gather_work=gather_work,
                            future=outer_future,
                            dequantize_reduce=dequantize_reduce_feedback,
                            update_feedback=update_feedback,
                            advance_policy=lambda: feedback_policy.advance(key),
                            completion_manager=active_completion_manager,
                            synchronize_completion=synchronize_async_feedback_completion,
                        ).run()

                    def complete(_ignored: Any = None) -> Any:
                        gathered = gather_work.wait()
                        restored = dequantize_reduce_tensors(
                            [_payload_buffer(payload) for payload in gathered.payloads],
                            tuple(prepared.shape),
                            config,
                            dtype=active_dtype,
                            extension_status=extension_status,
                            reduce=reduce,
                        )
                        if feedback_decision.update:
                            feedback.update(key, original=prepared, transmitted=restored)
                        feedback_policy.advance(key)
                        outer_future.set_result(restored)
                        return restored

                    inner_future = gather_work.get_future()
                    if inner_future is not None and hasattr(inner_future, "then"):
                        inner_future.then(complete)
                    else:
                        complete()
                    return outer_future
                gathered = active_all_gather(local_payload)
                restored = dequantize_reduce_tensors(
                    [_payload_buffer(payload) for payload in gathered.payloads],
                    tuple(prepared.shape),
                    config,
                    dtype=active_dtype,
                    extension_status=extension_status,
                    reduce=reduce,
                )
                if feedback_decision.update:
                    feedback.update(key, original=prepared, transmitted=restored)
                feedback_policy.advance(key)
                return restored
            collective = CompressedAllGatherReduce(
                config=config,
                compress=lambda tensor, active_config: _coerce_payload(
                    active_quantize(tensor, active_config),
                    shape=tuple(tensor.shape),
                    dtype=active_dtype,
                ),
                all_gather=active_all_gather,
                decompress=active_dequantize,
            )
            restored = collective.run(prepared, shape=tuple(prepared.shape), dtype=active_dtype, reduce=reduce)
            if feedback_decision.update:
                feedback.update(key, original=prepared, transmitted=restored)
            feedback_policy.advance(key)
            return restored

    elif effective_strategy == "all_reduce":
        if all_reduce is None:
            raise UnsupportedCollective(
                "all_reduce",
                reason="compressed all_reduce requires an explicit capability-bearing transport",
            )
        require_compressed_transport(
            all_reduce,
            collective="all_reduce",
            config=config,
            dtype=None if dtype == "auto" else dtype,
            output_layout="full",
        )
        processor = DDPBucketProcessor(
            config=config,
            quantize=active_quantize,
            dequantize=active_dequantize,
            all_reduce=all_reduce,
            error_feedback=feedback,
        )

        def process_bucket(bucket: Any) -> Any:
            tensor = bucket.buffer()
            if not _should_compress(tensor, min_numel=min_compress_numel):
                return native_all_reduce(_clone_tensor(tensor), reduce)
            require_compressed_transport(
                all_reduce,
                collective="all_reduce",
                config=config,
                dtype=_resolve_dtype(dtype, tensor),
                output_layout="full",
            )
            return processor.process(bucket, dtype=_resolve_dtype(dtype, tensor))

    else:
        raise ValueError(f"unsupported DDP comm hook strategy: {strategy}")

    def hook(state: Any, bucket: Any) -> Any:
        result = process_bucket(bucket)
        if hasattr(result, "set_result"):
            return result
        future = future_factory()
        future.set_result(result)
        return future

    hook._ccdl_strategy_plan = strategy_plan
    hook._ccdl_effective_strategy = effective_strategy
    _apply_ddp_annotations(hook, annotation_provider)
    return hook


def _resolve_dtype(dtype: str, tensor: Any) -> str:
    if dtype != "auto":
        return dtype
    tensor_dtype = str(getattr(tensor, "dtype", ""))
    if "bfloat16" in tensor_dtype:
        return "bf16"
    if "float16" in tensor_dtype or tensor_dtype.endswith("half"):
        return "fp16"
    if "float32" in tensor_dtype or tensor_dtype.endswith("float"):
        return "fp32"
    raise ValueError(f"cannot infer CCDL dtype from bucket tensor dtype: {tensor_dtype!r}")


def _coerce_payload(value: Any, *, shape: tuple[int, ...], dtype: str) -> CompressedPayload:
    if isinstance(value, CompressedPayload):
        return value
    if isinstance(value, dict) and "buffer" in value:
        return CompressedPayload(
            buffer=value["buffer"],
            shape=tuple(value.get("shape", shape)),
            dtype=str(value.get("dtype", dtype)),
            metadata=dict(value.get("metadata", {})),
        )
    return CompressedPayload(buffer=value, shape=shape, dtype=dtype)


def _payload_buffer(payload: Any) -> Any:
    return payload.buffer if hasattr(payload, "buffer") else payload


def _should_compress(tensor: Any, *, min_numel: int) -> bool:
    return min_numel <= 0 or _numel(tensor) >= min_numel


def _numel(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        return int(numel())
    total = 1
    for dim in getattr(tensor, "shape", ()):
        total *= int(dim)
    return total


def _clone_tensor(tensor: Any) -> Any:
    clone = getattr(tensor, "clone", None)
    return clone() if callable(clone) else tensor


def _reshape_to_shape(tensor: Any, shape: tuple[int, ...]) -> Any:
    if not hasattr(tensor, "reshape"):
        return tensor
    original_numel = 1
    for dim in shape:
        original_numel *= int(dim)
    flattened = tensor.reshape((-1,))
    try:
        trimmed = flattened[:original_numel]
    except TypeError:
        trimmed = flattened
    return trimmed.reshape(shape)


def _distributed_world_size(*, default: int) -> int:
    try:
        dist = import_module("torch.distributed")
        if hasattr(dist, "is_available") and not dist.is_available():
            return default
        if hasattr(dist, "is_initialized") and not dist.is_initialized():
            return default
        return int(dist.get_world_size())
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
        return default


def _distributed_rank(*, default: int) -> int:
    try:
        dist = import_module("torch.distributed")
        if hasattr(dist, "is_available") and not dist.is_available():
            return default
        if hasattr(dist, "is_initialized") and not dist.is_initialized():
            return default
        return int(dist.get_rank())
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
        return default


def _env_int(name: str) -> int | None:
    try:
        import os

        value = os.environ.get(name)
        return int(value) if value is not None and value != "" else None
    except ValueError:
        return None


def _apply_ddp_annotations(hook: Callable[[Any, Any], Any], provider: Callable[[], dict[str, Any]] | None) -> None:
    if provider is None:
        provider = _torch_ddp_annotations
    try:
        hook.__annotations__ = provider()
    except (ImportError, ModuleNotFoundError, AttributeError):
        return


def _torch_ddp_annotations() -> dict[str, Any]:
    torch = import_module("torch")
    dist = import_module("torch.distributed")
    return {
        "state": object,
        "bucket": dist.GradBucket,
        "return": torch.futures.Future[torch.Tensor],
    }
