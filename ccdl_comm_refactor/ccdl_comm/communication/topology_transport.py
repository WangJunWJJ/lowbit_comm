from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.collectives.reduce_scatter import ReducedShard
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor


def make_native_topology_all_reduce(
    *,
    method: str | None = None,
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
        active_method = method or _select_topology_method(world_size)
        output = tensor.clone() if callable(getattr(tensor, "clone", None)) else tensor
        if op == "mean" and world_size > 1:
            output = output / world_size
            op = "sum"
        if world_size <= 1:
            if async_op:
                return _TopologyWork(None, output, active_method)
            return output
        if active_method == "overlap-gather":
            work = _overlap_gather_all_reduce(
                output,
                op=op,
                config=config,
                dtype=dtype,
                extension_status=extension_status,
                dist=dist,
                quantize=quantize,
                dequantize=dequantize,
            )
            return work if async_op else work.wait()
        if active_method == "overlap-p2p":
            work = _overlap_p2p_all_reduce(
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
            return work if async_op else work.wait()
        if active_method == "overlap-tree":
            work = _overlap_tree_all_reduce(
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
            return work if async_op else work.wait()
        if active_method == "overlap-scale":
            work = _overlap_scale_all_reduce(
                output,
                config=config,
                dist=dist,
                import_module_fn=import_module_fn,
            )
            return work if async_op else work.wait()
        if active_method == "tree":
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
        elif active_method == "ring":
            _ring_all_reduce(
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
        elif active_method == "p2p":
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
            raise UnsupportedCollective(f"topology_all_reduce:{active_method}", reason="unsupported topology method")
        if async_op:
            return _TopologyWork(None, output, active_method)
        return output

    return transport


def make_legacy_topology_all_reduce(
    *,
    import_module_fn: Callable[[str], Any] = import_module,
    method: str | None = None,
) -> Callable[..., Any]:
    """Compatibility alias for the migrated native topology transport."""

    return make_native_topology_all_reduce(import_module_fn=import_module_fn, method=method)


def make_native_topology_reduce_scatter_shard(
    *,
    method: str | None = None,
    import_module_fn: Callable[[str], Any] = import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize: Callable[..., Any] = dequantize_tensor,
) -> Callable[..., ReducedShard]:
    """Create a native topology-aware reduce-scatter transport returning a local shard."""

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> ReducedShard | _TopologyWork:
        if op not in {"sum", "mean"}:
            raise UnsupportedCollective(f"topology_reduce_scatter:{op}", reason="only sum and mean are supported")
        requested_reduce = op
        dist = _distributed(import_module_fn)
        world_size = int(dist.get_world_size())
        active_method = method or _select_reduce_scatter_method(world_size)
        flat = tensor.reshape((-1,))
        original_numel = int(flat.numel())
        if original_numel % world_size != 0:
            raise UnsupportedCollective(
                f"topology_reduce_scatter:{active_method}",
                reason="native topology reduce-scatter requires tensor numel divisible by world size",
            )
        shard_numel = original_numel // world_size
        chunks = list(flat.chunk(world_size))
        rank = int(dist.get_rank())
        output = chunks[rank].clone() if callable(getattr(chunks[rank], "clone", None)) else chunks[rank]
        if op == "mean" and world_size > 1:
            for index, chunk in enumerate(chunks):
                chunks[index] = chunk / world_size
            output = output / world_size
            op = "sum"
        if world_size > 1:
            if active_method == "ring":
                _ring_reduce_scatter(
                    output,
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
            elif active_method == "p2p":
                _p2p_reduce_scatter(
                    output,
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
            else:
                raise UnsupportedCollective(
                    f"topology_reduce_scatter:{active_method}",
                    reason="only ring and p2p methods are supported",
                )
        reduced = ReducedShard(
            shard=output,
            shard_index=rank,
            shard_numel=shard_numel,
            original_shape=tuple(tensor.shape),
            original_numel=original_numel,
            world_size=world_size,
            reduce=requested_reduce,
            padded_numel=original_numel,
            dtype=dtype,
            transport=f"topology_{active_method}",
            metadata={
                "compression_bit": config.bit,
                "group_size": config.group_size,
                "method": active_method,
            },
        )
        if async_op:
            return _TopologyWork(None, reduced, active_method)
        return reduced

    return transport


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


class _CallbackTopologyWork:
    def __init__(self, work: Any, result: Any, method: str, complete: Callable[[], None]) -> None:
        self._work = work
        self._result = result
        self.method = method
        self._complete = complete
        self._completed = False

    def wait(self) -> Any:
        if not self._completed:
            wait = getattr(self._work, "wait", None)
            if callable(wait):
                wait()
            self._complete()
            self._completed = True
        return self._result


def _select_topology_method(world_size: int) -> str:
    if world_size <= 1:
        return "gather"
    if world_size == 2:
        return "tree"
    return "ring"


def _select_reduce_scatter_method(world_size: int) -> str:
    if world_size <= 2:
        return "p2p"
    return "ring"


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


def _ring_all_reduce(
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
        raise UnsupportedCollective("topology_all_reduce:ring", reason="ring requires tensor numel divisible by world size")
    chunks = list(flattened.chunk(world_size))
    rank = int(dist.get_rank())
    _ring_reduce_scatter(
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


def _overlap_gather_all_reduce(
    tensor: Any,
    *,
    op: str,
    config: CompressionConfig,
    dtype: str,
    extension_status: Any | None,
    dist: Any,
    quantize: Callable[..., Any],
    dequantize: Callable[..., Any],
) -> _CallbackTopologyWork:
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    shape = tuple(tensor.shape)
    q = quantize(tensor, config, extension_status=extension_status)
    q_buffer = q.new_empty((int(q.numel()) * world_size,), dtype=q.dtype, device=q.device)
    q_list = [q_buffer[int(q.numel()) * index : int(q.numel()) * (index + 1)] for index in range(world_size)]
    work = dist.all_gather_into_tensor(q_buffer, q, async_op=True)
    dequantize(q, shape, config, dtype=dtype, extension_status=extension_status, output=tensor, reduce_op="none")

    def complete() -> None:
        for remote_rank, remote_q in enumerate(q_list):
            if remote_rank == rank:
                continue
            dequantize(
                remote_q,
                shape,
                config,
                dtype=dtype,
                extension_status=extension_status,
                output=tensor,
                reduce_op=op,
            )

    return _CallbackTopologyWork(work, tensor, "overlap-gather", complete)


def _overlap_p2p_all_reduce(
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
) -> _CallbackTopologyWork:
    world_size = int(dist.get_world_size())
    flattened = tensor.reshape((-1,))
    if int(flattened.numel()) % world_size != 0:
        raise UnsupportedCollective(
            "topology_all_reduce:overlap-p2p",
            reason="overlap-p2p requires tensor numel divisible by world size",
        )
    chunks = list(flattened.chunk(world_size))
    rank = int(dist.get_rank())
    _ring_reduce_scatter(
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
    q = quantize(chunks[rank], config, extension_status=extension_status)
    q_buffer = q.new_empty((int(q.numel()) * world_size,), dtype=q.dtype, device=q.device)
    q_list = [q_buffer[int(q.numel()) * index : int(q.numel()) * (index + 1)] for index in range(world_size)]
    work = dist.all_gather_into_tensor(q_buffer, q, async_op=True)
    dequantize(
        q,
        tuple(chunks[rank].shape),
        config,
        dtype=dtype,
        extension_status=extension_status,
        output=chunks[rank],
        reduce_op="none",
    )

    def complete() -> None:
        for remote_rank, remote_q in enumerate(q_list):
            if remote_rank == rank:
                continue
            dequantize(
                remote_q,
                tuple(chunks[remote_rank].shape),
                config,
                dtype=dtype,
                extension_status=extension_status,
                output=chunks[remote_rank],
                reduce_op="none",
            )

    return _CallbackTopologyWork(work, tensor, "overlap-p2p", complete)


def _overlap_tree_all_reduce(
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
) -> _CallbackTopologyWork:
    world_size = int(dist.get_world_size())
    if world_size != 2 ** (world_size.bit_length() - 1):
        raise UnsupportedCollective("topology_all_reduce:overlap-tree", reason="tree requires power-of-two world size")
    rank = int(dist.get_rank())
    index2rank = _process_group_ranks(dist)
    torch = import_module_fn("torch")
    shape = tuple(tensor.shape)
    offset = 1
    final_works = None
    final_recv_q = None
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
        if offset * 2 >= world_size:
            final_works = works
            final_recv_q = recv_q
            break
        for work in works:
            work.wait()
        dequantize(recv_q, shape, config, dtype=dtype, extension_status=extension_status, output=tensor, reduce_op=op)
        offset *= 2

    def complete() -> None:
        if final_works is not None:
            for work in final_works:
                work.wait()
        if final_recv_q is not None:
            dequantize(
                final_recv_q,
                shape,
                config,
                dtype=dtype,
                extension_status=extension_status,
                output=tensor,
                reduce_op=op,
            )

    return _CallbackTopologyWork(None, tensor, "overlap-tree", complete)


def _overlap_scale_all_reduce(
    tensor: Any,
    *,
    config: CompressionConfig,
    dist: Any,
    import_module_fn: Callable[[str], Any],
) -> _CallbackTopologyWork:
    torch = import_module_fn("torch")
    group_size = int(config.group_size)
    flat = tensor.reshape((-1,))
    if int(flat.numel()) % group_size != 0:
        raise UnsupportedCollective(
            "topology_all_reduce:overlap-scale",
            reason="overlap-scale requires tensor numel divisible by group_size",
        )
    grouped = flat.reshape((-1, group_size))
    scale = grouped.abs().max(dim=-1, keepdim=True).values / 127
    dist.all_reduce(scale, op=dist.ReduceOp.MAX, async_op=False)
    q = (grouped / scale).to(torch.int8)
    work = dist.all_reduce(q, op=dist.ReduceOp.SUM, async_op=True)

    def complete() -> None:
        torch.mul(q, scale, out=grouped)

    return _CallbackTopologyWork(work, tensor, "overlap-scale", complete)


def _ring_reduce_scatter(
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
    if len(input_list) != world_size:
        raise UnsupportedCollective("topology_all_reduce:ring", reason="ring input chunks must match world size")
    rank = int(dist.get_rank())
    index2rank = _process_group_ranks(dist)
    torch = import_module_fn("torch")
    data_rank = list(range(world_size))
    for _round in range(world_size - 1):
        data_rank = [(index + 1) % world_size for index in data_rank]
        send_index = recv_index = send_target = recv_source = None
        for data_index, source in enumerate(data_rank):
            target = (source + 1) % world_size
            if source == rank:
                send_index = data_index
                send_target = target
            if target == rank:
                recv_index = data_index
                recv_source = source
        if send_index is None or recv_index is None or send_target is None or recv_source is None:
            raise UnsupportedCollective("topology_all_reduce:ring", reason="failed to plan ring step")
        q = quantize(input_list[send_index], config, extension_status=extension_status)
        recv_q = torch.empty_like(q)
        send_peer = index2rank[send_target]
        recv_peer = index2rank[recv_source]
        if rank % 2 == 0:
            ops = [
                dist.P2POp(dist.isend, q, send_peer),
                dist.P2POp(dist.irecv, recv_q, recv_peer),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv_q, recv_peer),
                dist.P2POp(dist.isend, q, send_peer),
            ]
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()
        dequantize(
            recv_q,
            tuple(input_list[recv_index].shape),
            config,
            dtype=dtype,
            extension_status=extension_status,
            output=input_list[recv_index],
            reduce_op=op,
        )
    copy = getattr(output, "copy_", None)
    if callable(copy):
        copy(input_list[rank], non_blocking=True)


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
