"""Backend-neutral reduction semantics shared by CCDL execution paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ReductionOperation = Literal["sum", "mean"]
TransportReduction = Literal["sum", "mean"]


@dataclass(frozen=True, slots=True)
class ReductionContract:
    """Describe the requested reduction and the transport's output semantics."""

    op: ReductionOperation
    world_size: int
    transport_output: TransportReduction = "sum"

    def __post_init__(self) -> None:
        if self.op not in {"sum", "mean"}:
            raise ValueError("op must be 'sum' or 'mean'")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int) or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if self.transport_output not in {"sum", "mean"}:
            raise ValueError("transport_output must be 'sum' or 'mean'")
        if self.op == "sum" and self.transport_output == "mean":
            raise ValueError("transport mean cannot satisfy sum without the original world-size-scaled values")

    @property
    def transport_op(self) -> ReductionOperation:
        """Return the operation the transport must execute."""

        return self.transport_output

    def normalize(self, tensor: Any) -> Any:
        """Normalize a transport result exactly once for the requested operation."""

        if self.op == "mean" and self.transport_output == "sum":
            return tensor / self.world_size
        return tensor
