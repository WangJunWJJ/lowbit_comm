"""Backend capability and lowering protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
)
from lowbit_comm.api.policy import (
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
)
from lowbit_comm.core.errors import CompileError

if TYPE_CHECKING:
    from lowbit_comm.runtime.work import CommunicationWork


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """One exact backend capability advertised to the compiler."""

    backend_id: str
    compression: CompressionKind
    collective: CollectiveKind
    topology: TopologyKind
    output: OutputSemantics
    min_world_size: int
    max_world_size: int | None
    supported_dtypes: frozenset[str]
    supports_async: bool

    def __post_init__(self) -> None:
        _validate_capability_fields(self)

    def supports(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> bool:
        """Return whether this capability can lower the exact request."""
        if type(intent) is not CommunicationIntent:
            return False
        if type(strategy) is not StrategySpec:
            return False
        return (
            self.compression is strategy.compression
            and self.collective is strategy.collective
            and self.topology is strategy.topology
            and self.output is intent.output
            and self.min_world_size <= intent.world_size
            and (self.max_world_size is None
                 or intent.world_size <= self.max_world_size)
            and intent.tensor.dtype in self.supported_dtypes
            and (self.supports_async
                 or intent.completion is not CompletionMode.ASYNC)
        )


class BackendPlan(Protocol):
    """A compiled backend operation ready for execution."""

    def execute(self, value: object) -> CommunicationWork[object]:
        """Execute the compiled operation for one input value."""


class Backend(Protocol):
    """A backend that advertises capabilities and lowers exact strategies."""

    @property
    def backend_id(self) -> str:
        """Return this backend's stable identifier."""

    def capabilities(self) -> tuple[BackendCapability, ...]:
        """Return immutable capability declarations."""

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> BackendPlan:
        """Lower the exact request into an executable backend plan."""


def _validate_capability_fields(capability: BackendCapability) -> None:
    """Reject non-deterministic values from a capability key."""
    if type(capability.backend_id) is not str:
        raise CompileError("Backend identifier must be a string.")
    if type(capability.compression) is not CompressionKind:
        raise CompileError("Capability compression must be CompressionKind.")
    if type(capability.collective) is not CollectiveKind:
        raise CompileError("Capability collective must be CollectiveKind.")
    if type(capability.topology) is not TopologyKind:
        raise CompileError("Capability topology must be TopologyKind.")
    if type(capability.output) is not OutputSemantics:
        raise CompileError("Capability output must be OutputSemantics.")
    if type(capability.min_world_size) is not int:
        raise CompileError("Capability minimum world size must be an integer.")
    if capability.min_world_size <= 0:
        raise CompileError("Capability minimum world size must be positive.")
    if capability.max_world_size is not None and (
        type(capability.max_world_size) is not int
        or capability.max_world_size < capability.min_world_size
    ):
        raise CompileError("Capability maximum world size is invalid.")
    if type(capability.supported_dtypes) is not frozenset or not all(
        type(dtype) is str for dtype in capability.supported_dtypes
    ):
        raise CompileError("Capability dtypes must be a frozenset of strings.")
    if type(capability.supports_async) is not bool:
        raise CompileError("Capability async support must be a boolean.")
