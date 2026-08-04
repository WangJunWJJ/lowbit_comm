from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ccdl_comm.config import CompressionConfig


@dataclass(frozen=True)
class CompressedPayload:
    """Compressed tensor payload plus reconstruction metadata."""

    buffer: Any
    shape: tuple[int, ...]
    dtype: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_buffer(self, buffer: Any) -> CompressedPayload:
        """Return a copy carrying a different compressed buffer."""

        return CompressedPayload(buffer=buffer, shape=self.shape, dtype=self.dtype, metadata=self.metadata)


@dataclass(frozen=True)
class CompressedAllReduce:
    """Adapter-level compressed all-reduce orchestration.

    The transport callable owns the actual distributed operation.  This class
    only defines the order of compression, transport, and reconstruction.
    """

    config: CompressionConfig
    compress: Callable[[Any, CompressionConfig], CompressedPayload]
    all_reduce: Callable[[CompressedPayload, str], CompressedPayload]
    decompress: Callable[[CompressedPayload, CompressionConfig], Any]

    def run(self, tensor: Any, *, op: str = "sum") -> Any:
        payload = self.compress(tensor, self.config)
        reduced = self.all_reduce(payload, op)
        return self.decompress(reduced, self.config)
