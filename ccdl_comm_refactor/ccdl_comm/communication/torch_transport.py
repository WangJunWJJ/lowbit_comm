from __future__ import annotations

from collections.abc import Callable
from importlib import import_module as _import_module
from typing import Any

from .collectives import CompressedPayload


class TorchDistributedUnavailableError(RuntimeError):
    """Raised when torch.distributed cannot run the requested transport."""


def _reduce_op(dist: Any, op: str) -> Any:
    normalized = op.strip().upper()
    reduce_op = getattr(dist, "ReduceOp", None)
    if reduce_op is None or not hasattr(reduce_op, normalized):
        raise ValueError(f"unsupported torch.distributed reduce op: {op}")
    return getattr(reduce_op, normalized)


def make_torch_all_reduce(
    *,
    import_module: Callable[[str], Any] = _import_module,
) -> Callable[[CompressedPayload, str], CompressedPayload]:
    """Create an all-reduce transport backed by ``torch.distributed``."""

    def transport(payload: CompressedPayload, op: str) -> CompressedPayload:
        try:
            dist = import_module("torch.distributed")
        except (ImportError, ModuleNotFoundError) as exc:
            raise TorchDistributedUnavailableError("torch.distributed is not available") from exc

        if not dist.is_available() or not dist.is_initialized():
            raise TorchDistributedUnavailableError("torch.distributed is not initialized")

        dist.all_reduce(payload.buffer, op=_reduce_op(dist, op))
        return payload

    return transport
