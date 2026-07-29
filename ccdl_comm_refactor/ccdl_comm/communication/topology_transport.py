from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor


def make_native_topology_all_reduce(
    *,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize: Callable[..., Any] = dequantize_tensor,
) -> Callable[..., Any]:
    """Create a native topology-aware all-reduce transport.

    The algorithms mirror the pre-refactor CCDL tree and p2p communication
    shapes, but use `ccdl_comm` quantization/dequantization and do not import the
    legacy `ccdl` package.
    """

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> Any:
        if op not in {"sum", "mean"}:
            raise UnsupportedCollective(f"topology_all_reduce:{op}", reason="only sum and mean are supported")
        dist = _distributed(import_module_fn)
        world_size = int(dist.get_world_size())
        method = _select_topology_method(world_size)
        output = tensor.clone() if callable(getattr(tensor, "clone", None)) else tensor
        if op == "mean" and world_size > 1:
            output = output / world_size
            op = "sum"
        if world_size <= 1:
            if async_op:
                return _TopologyWork(None, output, method)
            return output
        if method == "tree":
            _tree_all_reduce(
                output,
                op=op,
                config=config,
                dtype=dtype,
                extension_status=extension_status,
                dist=dist,
                import_module_fn=import_module_fn,
                quantize=quantize,
                dequantize=dequantize,
            )
        elif method == "p2p":
            _p2p_all_reduce(
                output,
                op=op,
                config=config,
                dtype=dtype,
                extension_status=extension_status,
                dist=dist,
                import_module_fn=import_module_fn,
                quantize=quantize,
                dequantize=dequantize,
            )
        else:
            raise UnsupportedCollective(f"topology_all_reduce:{method}", reason="unsupported topology method")
        if async_op:
            return _TopologyWork(None, output, method)
        return output

    return transport


def make_legacy_topology_all_reduce(
    *,
    import_module_fn: Callable[[str], Any] = import_module,
) -> Callable[..., Any]:
    """Compatibility alias for the migrated native topology transport."""

    return make_native_topology_all_reduce(import_module_fn=import_module_fn)


def make_legacy_bridge_topology_all_reduce(
    *,
    import_module_fn: Callable[[str], Any] = import_module,
) -> Callable[..., Any]:
    """Create the original bridge transport backed by the legacy `ccdl` package."""

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> Any:
        del extension_status
        if op not in {"sum", "mean"}:
            raise UnsupportedCollective(f"topology_all_reduce:{op}", reason="only sum and mean are supported")
        ccdl_comm = import_module_fn("ccdl.comm")
        quantization = import_module_fn("ccdl.quantization")
        dist = _distributed(import_module_fn)
        quantizer = quantization.Quantizer(
            config.group_size,
            -1,
            config.bit,
            config.topk,
            config.stochastic,
            dtype,
            quant_type=config.quant_type,
            compact=config.compact,
        )
        method = _select_topology_method(int(dist.get_world_size()))
        output = tensor.clone() if callable(getattr(tensor, "clone", None)) else tensor
        work = ccdl_comm.qall_reduce(
            output,
            op=op,
            quantizer=quantizer,
            method=method,
            keep_self=False,
            async_op=async_op,
        )
        if async_op:
            return _TopologyWork(work, output, method)
        return output

    return transport


class _TopologyWork:
    def __init__(self, work: Any, result: Any, method: str) -> None:
        self._work = work
        self._result = result
        self.method = method

    def wait(self) -> Any:
        wait = getattr(self._work, "wait", None)
        if callable(wait):
            wait()
        return self._result


def _select_topology_method(world_size: int) -> str:
    if world_size <= 1:
        return "gather"
    if world_size == 2:
        return "tree"
    return "p2p"


def _tree_all_reduce(
    tensor: Any,
    *,
    op: str,
    config: CompressionConfig,
    dtype: str,
    extension_status: Any | None,
    dist: Any,
    import_module_fn: Callable[[str], Any],
    quantize: Callable[..., Any],
    dequantize: Callable[..., Any],
) -> None:
    world_size = int(dist.get_world_size())
    if world_size != 2 ** (world_size.bit_length() - 1):
        raise UnsupportedCollective("topology_all_reduce:tree", reason="tree requires power-of-two world size")
    rank = int(dist.get_rank())
    index2rank = _process_group_ranks(dist)
    torch = import_module_fn("torch")
    offset = 1
    shape = tuple(tensor.shape)
    while offset < world_size:
        q = quantize(tensor, config, extension_status=extension_status)
        recv_q = torch.empty_like(q)
        if (rank // offset) % 2 == 0:
            target_rank = index2rank[rank + offset]
            ops = [
                dist.P2POp(dist.isend, q, target_rank),
                dist.P2POp(dist.irecv, recv_q, target_rank),
            ]
        else:
            target_rank = index2rank[rank - offset]
            ops = [
                dist.P2POp(dist.irecv, recv_q, target_rank),
                dist.P2POp(dist.isend, q, target_rank),
            ]
        works = dist.batch_isend_irecv(ops)
        dequantize(q, shape, config, dtype=dtype, extension_status=extension_status, output=tensor, reduce_op="none")
        for work in works:
            work.wait()
        dequantize(recv_q, shape, config, dtype=dtype, extension_status=extension_status, output=tensor, reduce_op=op)
        offset *= 2


def _p2p_all_reduce(
    tensor: Any,
    *,
    op: str,
    config: CompressionConfig,
    dtype: str,
    extension_status: Any | None,
    dist: Any,
    import_module_fn: Callable[[str], Any],
    quantize: Callable[..., Any],
    dequantize: Callable[..., Any],
) -> None:
    world_size = int(dist.get_world_size())
    flattened = tensor.reshape((-1,))
    if int(flattened.numel()) % world_size != 0:
        raise UnsupportedCollective("topology_all_reduce:p2p", reason="p2p requires tensor numel divisible by world size")
    chunks = list(flattened.chunk(world_size))
    rank = int(dist.get_rank())
    _p2p_reduce_scatter(
        chunks[rank],
        chunks,
        op=op,
        config=config,
        dtype=dtype,
        extension_status=extension_status,
        dist=dist,
        import_module_fn=import_module_fn,
        quantize=quantize,
        dequantize=dequantize,
    )
    _qall_gather_base(
        chunks,
        chunks[rank],
        config=config,
        dtype=dtype,
        extension_status=extension_status,
        dist=dist,
        import_module_fn=import_module_fn,
        quantize=quantize,
        dequantize=dequantize,
    )


def _p2p_reduce_scatter(
    output: Any,
    input_list: list[Any],
    *,
    op: str,
    config: CompressionConfig,
    dtype: str,
    extension_status: Any | None,
    dist: Any,
    import_module_fn: Callable[[str], Any],
    quantize: Callable[..., Any],
    dequantize: Callable[..., Any],
) -> None:
    world_size = int(dist.get_world_size())
    if world_size <= 0 or world_size & (world_size - 1):
        raise UnsupportedCollective("topology_all_reduce:p2p", reason="p2p requires power-of-two world size")
    rank = int(dist.get_rank())
    index2rank = _process_group_ranks(dist)
    torch = import_module_fn("torch")
    for offset in range(1, world_size):
        target = rank ^ offset
        q = quantize(input_list[target], config, extension_status=extension_status)
        recv_q = torch.empty_like(q)
        target_rank = index2rank[target]
        if rank < target:
            ops = [
                dist.P2POp(dist.isend, q, target_rank),
                dist.P2POp(dist.irecv, recv_q, target_rank),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv_q, target_rank),
                dist.P2POp(dist.isend, q, target_rank),
            ]
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        dequantize(
            recv_q,
            tuple(input_list[rank].shape),
            config,
            dtype=dtype,
            extension_status=extension_status,
            output=input_list[rank],
            reduce_op=op,
        )
    copy = getattr(output, "copy_", None)
    if callable(copy):
        copy(input_list[rank], non_blocking=True)


def _qall_gather_base(
    tensor_list: list[Any],
    tensor: Any,
    *,
    config: CompressionConfig,
    dtype: str,
    extension_status: Any | None,
    dist: Any,
    import_module_fn: Callable[[str], Any],
    quantize: Callable[..., Any],
    dequantize: Callable[..., Any],
) -> None:
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    if world_size <= 1:
        return
    q = quantize(tensor, config, extension_status=extension_status)
    q_buffer = q.new_empty((int(q.numel()) * world_size,), dtype=q.dtype, device=q.device)
    q_list = [q_buffer[int(q.numel()) * index : int(q.numel()) * (index + 1)] for index in range(world_size)]
    dist.all_gather_into_tensor(q_buffer, q)
    for remote_rank, remote_q in enumerate(q_list):
        dequantize(
            remote_q,
            tuple(tensor_list[remote_rank].shape),
            config,
            dtype=dtype,
            extension_status=extension_status,
            output=tensor_list[remote_rank],
            reduce_op="none",
        )


def _process_group_ranks(dist: Any) -> list[int]:
    getter = getattr(dist, "get_process_group_ranks", None)
    if callable(getter):
        try:
            return list(getter(None))
        except (KeyError, TypeError):
            pass
    return list(range(int(dist.get_world_size())))


def _distributed(import_module_fn: Callable[[str], Any]) -> Any:
    try:
        dist = import_module_fn("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError("torch.distributed is not available") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError("torch.distributed is not initialized")
    return dist
