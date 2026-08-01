from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor
from ccdl_comm.collectives.work import CollectiveWork, ImmediateWork
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.gather_reduce import CompressedAllGatherReduce, GatheredPayloads, _sum_tensors
from ccdl_comm.communication.payload_packing import (
    DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    make_fused_payload_all_gather,
    make_payload_all_gather,
    should_fuse_payload,
)
from ccdl_comm.communication.torch_transport import (
    make_torch_all_gather,
    make_torch_all_reduce,
    make_torch_async_all_gather,
    make_torch_async_all_reduce,
)
from ccdl_comm.communication.topology_transport import make_legacy_topology_all_reduce
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.executor import CompiledCommunicationPlan


_make_payload_all_gather = make_payload_all_gather


def _compile_cuda_shortcut(tensor: Any, **kwargs: Any) -> CompiledCommunicationPlan:
    from ccdl_comm.cuda.shortcut import compile_cuda_shortcut

    return compile_cuda_shortcut(tensor, **kwargs)


@dataclass(frozen=True)
class _DictPayload:
    buffer: Any
    shape: tuple[int, ...]
    dtype: str


def compressed_all_reduce(
    tensor: Any,
    *,
    config: CompressionConfig,
    op: str = "mean",
    strategy: str = "all_gather",
    async_op: bool = False,
    world_size: int | None = None,
    dtype: str = "auto",
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None,
    all_gather: Callable[[Any], GatheredPayloads] | None = None,
    topology_all_reduce: Callable[..., Any] | None = None,
    topology_method: str | None = None,
    fuse_payload: bool = False,
    fuse_payload_min_numel: int = DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    extension_status: CudaExtensionStatus | None = None,
    completion_manager: CudaCompletionManager | Any | None = None,
    compiled_plan: CompiledCommunicationPlan | None = None,
) -> Any | CollectiveWork[Any]:
    """Run a compressed all-reduce over a tensor.

    Args:
        tensor: Tensor-like object to reduce.
        config: Compression policy.
        op: Reduction operation. ``mean`` maps to transport ``sum`` followed by
            division by world size.
        strategy: Collective strategy. The first correctness-preserving
            production strategy is ``all_gather``. ``all_reduce`` is available
            for injected transports that understand the compressed payload.
        async_op: When true, return a work object with ``wait()``.
        world_size: Optional world size override for tests or custom runtimes.
        dtype: Source dtype name or ``auto`` to infer from the tensor.
        quantize: Optional injected quantizer for tests/custom runtimes.
        dequantize: Optional injected dequantizer for tests/custom runtimes.
        all_reduce: Optional injected transport.
        topology_method: Optional explicit topology method such as ``tree``,
            ``p2p``, or ``ring`` when ``strategy='topology'`` and the default
            topology transport is used.
        fuse_payload: Pack compressed buffer and tensor metadata into one
            byte all-gather when using the ``all_gather`` strategy.
        fuse_payload_min_numel: Minimum tensor elements required before
            enabling fused payload packing.
        extension_status: Optional preloaded CUDA extension status.

    Returns:
        The reduced tensor, or a ``CollectiveWork`` when ``async_op=True``.

    Raises:
        UnsupportedCollective: If ``strategy`` or ``op`` is unsupported.
    """

    if strategy not in {"all_gather", "all_reduce", "topology"}:
        raise UnsupportedCollective(
            f"all_reduce:{strategy}",
            reason="only strategy='all_gather', strategy='all_reduce', and strategy='topology' are implemented",
        )
    if op not in {"sum", "mean"}:
        raise UnsupportedCollective(f"all_reduce:{op}", reason="only op='sum' and op='mean' are implemented")

    if compiled_plan is not None:
        return _finish_compiled(compiled_plan.run(tensor), async_op=async_op)

    injected = any(
        value is not None
        for value in (quantize, dequantize, all_reduce, all_gather, topology_all_reduce, completion_manager)
    )
    compile_supported = (
        not injected
        and op == "mean"
        and strategy in {"all_gather", "topology"}
        and world_size is None
        and topology_method is None
        and not fuse_payload
    )
    if compile_supported:
        compiled = _compile_cuda_shortcut(
            tensor,
            collective="all_reduce",
            strategy=strategy,
            output_layout="full",
            config=config,
            async_op=async_op,
            dtype=dtype,
            extension_status=extension_status,
        )
        return _finish_compiled(compiled.run(tensor), async_op=async_op)

    return _run_compressed_all_reduce(
        tensor,
        config=config,
        op=op,
        strategy=strategy,
        async_op=async_op,
        world_size=world_size,
        dtype=dtype,
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=all_reduce,
        all_gather=all_gather,
        topology_all_reduce=topology_all_reduce,
        topology_method=topology_method,
        fuse_payload=fuse_payload,
        fuse_payload_min_numel=fuse_payload_min_numel,
        extension_status=extension_status,
        completion_manager=completion_manager,
    )


def _run_compressed_all_reduce(
    tensor: Any,
    *,
    config: CompressionConfig,
    op: str = "mean",
    strategy: str = "all_gather",
    async_op: bool = False,
    world_size: int | None = None,
    dtype: str = "auto",
    quantize: Callable[[Any, CompressionConfig], Any] | None = None,
    dequantize: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any] | None = None,
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload] | None = None,
    all_gather: Callable[[Any], GatheredPayloads] | None = None,
    topology_all_reduce: Callable[..., Any] | None = None,
    topology_method: str | None = None,
    fuse_payload: bool = False,
    fuse_payload_min_numel: int = DEFAULT_FUSED_PAYLOAD_MIN_NUMEL,
    extension_status: CudaExtensionStatus | None = None,
    completion_manager: CudaCompletionManager | Any | None = None,
    process_group: Any | None = None,
) -> Any | CollectiveWork[Any]:
    """Legacy implementation used by pre-bound CUDA executors and injections."""

    active_dtype = _resolve_dtype(dtype, tensor)
    shape = tuple(getattr(tensor, "shape", ()))
    active_quantize = quantize or _extension_quantize(extension_status)
    active_dequantize = dequantize or _extension_dequantize(extension_status)

    if strategy == "topology":
        active_topology_all_reduce = topology_all_reduce or make_legacy_topology_all_reduce(method=topology_method)
        return active_topology_all_reduce(
            tensor,
            config=config,
            op=op,
            async_op=async_op,
            dtype=active_dtype,
            extension_status=extension_status,
        )

    if strategy == "all_gather":
        if async_op and all_gather is None:
            local_payload = _coerce_payload(active_quantize(tensor, config), shape=shape, dtype=active_dtype)
            if not local_payload.metadata:
                gather_work = _make_group_transport(
                    make_torch_async_all_gather,
                    process_group,
                )(_payload_buffer(local_payload))
                manager = completion_manager or CudaCompletionManager()

                def complete_gather_reduce() -> Any:
                    payloads = _payloads_from_async_gather(gather_work, local_payload)
                    decoded = [active_dequantize(payload, shape, config, active_dtype) for payload in payloads]
                    reduced = _sum_tensors(decoded)
                    if op == "mean":
                        return reduced / int(gather_work.world_size)
                    return reduced

                return manager.create_work(
                    result=None,
                    handle=gather_work,
                    complete=complete_gather_reduce,
                    resources=(local_payload, *tuple(gather_work.payloads)),
                )
        active_all_gather = all_gather
        if active_all_gather is None:
            buffer_all_gather = _make_group_transport(make_torch_all_gather, process_group)
            active_all_gather = (
                make_fused_payload_all_gather(buffer_all_gather)
                if should_fuse_payload(tensor, enabled=fuse_payload, min_numel=fuse_payload_min_numel)
                else make_payload_all_gather(buffer_all_gather)
            )
        collective = CompressedAllGatherReduce(
            config=config,
            compress=active_quantize,
            all_gather=active_all_gather,
            decompress=active_dequantize,
        )
        restored = collective.run(tensor, shape=shape, dtype=active_dtype, reduce=op)
        if async_op:
            return ImmediateWork(restored)
        return restored

    payload = _coerce_payload(active_quantize(tensor, config), shape=shape, dtype=active_dtype)
    if async_op and all_reduce is None:
        reduce_work = _make_group_transport(
            make_torch_async_all_reduce,
            process_group,
        )(payload, "sum" if op == "mean" else op)
        manager = completion_manager or CudaCompletionManager()

        def complete_all_reduce() -> Any:
            restored = active_dequantize(reduce_work.payload, shape, config, active_dtype)
            if op == "mean":
                return restored / _resolve_world_size(world_size)
            return restored

        return manager.create_work(
            result=None,
            handle=reduce_work,
            complete=complete_all_reduce,
            resources=(payload, reduce_work.payload),
        )

    active_all_reduce = all_reduce or _make_group_transport(make_torch_all_reduce, process_group)
    reduced = active_all_reduce(payload, "sum" if op == "mean" else op)
    restored = active_dequantize(reduced, shape, config, active_dtype)
    if op == "mean":
        restored = restored / _resolve_world_size(world_size)
    if async_op:
        return ImmediateWork(restored)
    return restored


def _finish_compiled(work: Any, *, async_op: bool) -> Any:
    return work if async_op else work.wait()


def _make_group_transport(factory: Callable[..., Any], process_group: Any | None) -> Any:
    return factory() if process_group is None else factory(group=process_group)


def _extension_quantize(extension_status: CudaExtensionStatus | None) -> Callable[[Any, CompressionConfig], Any]:
    def quantize_with_extension(tensor: Any, config: CompressionConfig) -> CompressedPayload:
        return CompressedPayload(
            buffer=quantize_tensor(tensor, config, extension_status=extension_status),
            shape=tuple(getattr(tensor, "shape", ())),
            dtype=_resolve_dtype("auto", tensor),
        )

    return quantize_with_extension


def _extension_dequantize(extension_status: CudaExtensionStatus | None) -> Callable[[Any, tuple[int, ...], CompressionConfig, str], Any]:
    def dequantize_with_extension(payload: Any, shape: tuple[int, ...], config: CompressionConfig, dtype: str) -> Any:
        return dequantize_tensor(_payload_buffer(payload), shape, config, dtype=dtype, extension_status=extension_status)

    return dequantize_with_extension


def _coerce_payload(value: Any, *, shape: tuple[int, ...], dtype: str) -> CompressedPayload:
    if isinstance(value, CompressedPayload):
        return value
    if isinstance(value, dict) and "buffer" in value:
        return CompressedPayload(buffer=value["buffer"], shape=shape, dtype=dtype)
    return CompressedPayload(buffer=value, shape=shape, dtype=dtype)


def _payload_buffer(payload: Any) -> Any:
    return payload.buffer if hasattr(payload, "buffer") else payload


def _payloads_from_async_gather(work: Any, local_payload: CompressedPayload) -> list[CompressedPayload]:
    return [local_payload.with_buffer(buffer) for buffer in work.payloads]


def _resolve_world_size(world_size: int | None) -> int:
    if world_size is not None:
        return world_size
    dist = import_module("torch.distributed")
    return dist.get_world_size()


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
    raise ValueError(f"cannot infer CCDL dtype from tensor dtype: {tensor_dtype!r}")
