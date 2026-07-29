from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import TorchDistributedUnavailableError, UnsupportedCollective


def make_legacy_topology_all_reduce(
    *,
    import_module_fn: Callable[[str], Any] = import_module,
) -> Callable[..., Any]:
    """Create a topology-aware all-reduce transport backed by the legacy CCDL algorithms.

    This is the first migration bridge for the pre-refactor high-performance
    tree/p2p/ring implementations. It keeps the new `ccdl_comm` API surface while
    preserving the old optimized topology choices until native transports are
    fully reimplemented inside `ccdl_comm`.
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
        method = _select_legacy_method(int(dist.get_world_size()))
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


def _select_legacy_method(world_size: int) -> str:
    if world_size <= 1:
        return "gather"
    if world_size == 2:
        return "tree"
    return "p2p"


def _distributed(import_module_fn: Callable[[str], Any]) -> Any:
    try:
        dist = import_module_fn("torch.distributed")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchDistributedUnavailableError("torch.distributed is not available") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise TorchDistributedUnavailableError("torch.distributed is not initialized")
    return dist
