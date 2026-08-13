"""Backend extension protocol for lowering and executable binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .context import CompileContext, RuntimeBindings
from .lowered import LoweredProgram
from .program import CommunicationProgram


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    target: str
    supported_bits: frozenset[int]
    supported_algorithms: frozenset[str] = frozenset(
        {"native", "compressed_all_gather", "compressed_reduce_scatter", "compressed_rs_ag"}
    )
    supports_full_tensor: bool = True
    supports_reduced_shard: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("backend target must be a non-empty string")
        object.__setattr__(self, "supported_bits", frozenset(self.supported_bits))
        object.__setattr__(
            self,
            "supported_algorithms",
            frozenset(self.supported_algorithms),
        )


class CompiledExecutable(Protocol):
    def run(self, value: Any) -> Any: ...


class CommunicationBackend(Protocol):
    name: str
    abi_version: int

    def capabilities(self, context: CompileContext) -> BackendCapabilities: ...

    def lower(
        self,
        program: CommunicationProgram,
        context: CompileContext,
        bindings: RuntimeBindings,
    ) -> LoweredProgram: ...

    def compile(self, lowered: LoweredProgram) -> CompiledExecutable: ...
