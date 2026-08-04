from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module as _import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective
from ccdl_comm.quantization.codec import dequantize_reduce_tensors, quantize_tensor


class _GroupBoundDistributed:
    """Bind torch.distributed calls to one compile-time process group."""

    def __init__(self, distributed: Any, group: object) -> None:
        self._distributed = distributed
        self._group = group

    def __getattr__(self, name: str) -> Any:
        return getattr(self._distributed, name)

    def get_world_size(self, group: object | None = None) -> int:
        return int(self._distributed.get_world_size(group=self._bound(group)))

    def get_rank(self, group: object | None = None) -> int:
        return int(self._distributed.get_rank(group=self._bound(group)))

    def all_to_all(
        self,
        output: Any,
        input: Any,
        *,
        async_op: bool = False,
        group: object | None = None,
    ) -> Any:
        return self._distributed.all_to_all(
            output,
            input,
            async_op=async_op,
            group=self._bound(group),
        )

    def all_gather(
        self,
        output: Any,
        input: Any,
        *,
        async_op: bool = False,
        group: object | None = None,
    ) -> Any:
        return self._distributed.all_gather(
            output,
            input,
            async_op=async_op,
            group=self._bound(group),
        )

    def all_gather_into_tensor(
        self,
        output: Any,
        input: Any,
        *,
        async_op: bool = False,
        group: object | None = None,
    ) -> Any:
        return self._distributed.all_gather_into_tensor(
            output,
            input,
            async_op=async_op,
            group=self._bound(group),
        )

    def _bound(self, group: object | None) -> object:
        if group is not None and group is not self._group:
            raise ValueError("group-bound transport cannot override its process group")
        return self._group


def make_group_bound_importer(
    group: object,
    *,
    import_module: Callable[[str], Any] = _import_module,
) -> Callable[[str], Any]:
    """Return an importer whose distributed module uses a precreated group."""

    distributed = _GroupBoundDistributed(import_module("torch.distributed"), group)

    def import_bound(name: str) -> Any:
        return distributed if name == "torch.distributed" else import_module(name)

    return import_bound


@dataclass(frozen=True)
class _HierarchicalGroups:
    local_group_size: int
    local_groups: tuple[tuple[int, ...], ...]
    leader_ranks: tuple[int, ...]
    local_group: Any
    leader_group: Any | None
    local_ranks: tuple[int, ...]
    local_leader: int


def make_torch_hierarchical_all_reduce(
    *,
    local_group_size: int = 2,
    import_module: Callable[[str], Any] = _import_module,
    quantize: Callable[..., Any] = quantize_tensor,
    dequantize_reduce: Callable[..., Any] = dequantize_reduce_tensors,
) -> Callable[..., Any]:
    """Create a torch.distributed-backed hierarchical compressed all-reduce.

    The prototype uses compressed local-group gather/reduce, native leader
    all-reduce over restored partial tensors, then broadcasts the full result
    within each local group. It preserves full DDP bucket output semantics.
    """

    group_cache: dict[tuple[int, int], _HierarchicalGroups] = {}

    def transport(
        tensor: Any,
        *,
        config: CompressionConfig,
        op: str,
        async_op: bool,
        dtype: str,
        extension_status: Any | None,
    ) -> Any:
        if async_op:
            raise UnsupportedCollective("hierarchical:async", reason="hierarchical prototype is synchronous")
        if op not in {"sum", "mean"}:
            raise UnsupportedCollective(f"hierarchical:{op}", reason="only op='sum' and op='mean' are implemented")

        dist = _distributed(import_module)
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())
        groups = _get_or_create_groups(
            dist,
            rank=rank,
            world_size=world_size,
            local_group_size=local_group_size,
            cache=group_cache,
        )

        local_payload = quantize(tensor, config, extension_status=extension_status)
        gathered = [local_payload.new_empty(tuple(local_payload.shape)) for _ in groups.local_ranks]
        dist.all_gather(gathered, local_payload, group=groups.local_group)
        restored = dequantize_reduce(
            gathered,
            tuple(tensor.shape),
            config,
            dtype=dtype,
            extension_status=extension_status,
            reduce="sum",
        )
        if rank == groups.local_leader and groups.leader_group is not None:
            dist.all_reduce(restored, op=dist.ReduceOp.SUM, group=groups.leader_group)
        dist.broadcast(restored, src=groups.local_leader, group=groups.local_group)
        if op == "mean":
            restored /= world_size
        return restored

    return transport


def _distributed(import_module: Callable[[str], Any]) -> Any:
    try:
        dist = import_module("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError("torch.distributed is not available") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError("torch.distributed is not initialized")
    return dist


def _get_or_create_groups(
    dist: Any,
    *,
    rank: int,
    world_size: int,
    local_group_size: int,
    cache: dict[tuple[int, int], _HierarchicalGroups],
) -> _HierarchicalGroups:
    if local_group_size <= 0:
        raise UnsupportedCollective("hierarchical:local_group_size", reason="local_group_size must be positive")
    if world_size % local_group_size != 0:
        raise UnsupportedCollective(
            "hierarchical:world_size",
            reason="world_size must be divisible by local_group_size for the prototype",
        )
    key = (world_size, local_group_size)
    if key in cache:
        return cache[key]

    local_groups = tuple(
        tuple(range(start, start + local_group_size))
        for start in range(0, world_size, local_group_size)
    )
    leader_ranks = tuple(group[0] for group in local_groups)
    local_group_handles = {group: dist.new_group(list(group)) for group in local_groups}
    leader_group = dist.new_group(list(leader_ranks)) if len(leader_ranks) > 1 else None
    local_ranks = next(group for group in local_groups if rank in group)
    groups = _HierarchicalGroups(
        local_group_size=local_group_size,
        local_groups=local_groups,
        leader_ranks=leader_ranks,
        local_group=local_group_handles[local_ranks],
        leader_group=leader_group,
        local_ranks=local_ranks,
        local_leader=local_ranks[0],
    )
    cache[key] = groups
    return groups
