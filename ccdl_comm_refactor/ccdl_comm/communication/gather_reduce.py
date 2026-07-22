from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ccdl_comm.config import CompressionConfig


@dataclass(frozen=True)
class GatheredPayloads:
    """Compressed payloads gathered from all participating ranks."""

    payloads: Sequence[Any]
    world_size: int


@dataclass(frozen=True)
class CompressedAllGatherReduce:
    """Compressed gather-then-reduce adapter for DDP communication hooks."""

    config: CompressionConfig
    compress: Callable[[Any, CompressionConfig], Any]
    all_gather: Callable[[Any], GatheredPayloads]
    decompress: Callable[[Any, tuple[int, ...], CompressionConfig, str], Any]

    def run(self, tensor: Any, *, shape: tuple[int, ...], dtype: str, reduce: str = "mean") -> Any:
        local_payload = self.compress(tensor, self.config)
        gathered = self.all_gather(local_payload)
        decoded = [self.decompress(payload, shape, self.config, dtype) for payload in gathered.payloads]
        reduced = _sum_tensors(decoded)
        if reduce == "mean":
            return reduced / gathered.world_size
        if reduce == "sum":
            return reduced
        raise ValueError(f"unsupported gather-reduce mode: {reduce}")


def _sum_tensors(tensors: Sequence[Any]) -> Any:
    if not tensors:
        raise ValueError("cannot reduce an empty payload list")
    total = tensors[0]
    for tensor in tensors[1:]:
        total = total + tensor
    return total
