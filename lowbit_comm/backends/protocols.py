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
    StrategySpec,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.signatures import _require_dataclass_field_coverage
from lowbit_comm.core.validation import _fresh_validate_exact

if TYPE_CHECKING:
    from lowbit_comm.runtime.work import CommunicationWork


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """One exact backend capability advertised to the compiler."""

    backend_id: str
    strategy: StrategySpec
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
            self.strategy == strategy
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

    backend_id: str

    def capabilities(self) -> tuple[BackendCapability, ...]:
        """Return immutable capability declarations."""

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> BackendPlan:
        """Lower the exact request into an executable backend plan."""


_STRATEGY_SNAPSHOT_FIELDS = frozenset(
    {
        "compression",
        "collective",
        "topology",
        "group_size",
        "accumulation_dtype",
        "error_feedback",
        "parameter_error_feedback",
        "overlap",
        "workspace_budget_bytes",
    }
)
_CAPABILITY_SNAPSHOT_FIELDS = frozenset(
    {
        "backend_id",
        "strategy",
        "output",
        "min_world_size",
        "max_world_size",
        "supported_dtypes",
        "supports_async",
    }
)


def _snapshot_backend_capability(
    capability: object,
) -> BackendCapability:
    """Return a freshly validated, explicitly reconstructed capability."""
    source = _fresh_validate_exact(
        capability,
        BackendCapability,
        BackendCapability.__post_init__,
        "Backend capability snapshot source is invalid.",
    )
    _require_dataclass_field_coverage(
        source,
        BackendCapability,
        _CAPABILITY_SNAPSHOT_FIELDS,
        "Backend capability snapshot fields require explicit coverage.",
    )
    strategy = _snapshot_strategy(source.strategy)
    return BackendCapability(
        backend_id=source.backend_id,
        strategy=strategy,
        output=source.output,
        min_world_size=source.min_world_size,
        max_world_size=source.max_world_size,
        supported_dtypes=frozenset(
            dtype for dtype in source.supported_dtypes
        ),
        supports_async=source.supports_async,
    )


def _snapshot_strategy(strategy: object) -> StrategySpec:
    """Return an explicit independent snapshot of one exact strategy."""
    source = _fresh_validate_exact(
        strategy,
        StrategySpec,
        StrategySpec.__post_init__,
        "Backend capability strategy snapshot source is invalid.",
    )
    _require_dataclass_field_coverage(
        source,
        StrategySpec,
        _STRATEGY_SNAPSHOT_FIELDS,
        "Backend capability strategy snapshot fields require explicit "
        "coverage.",
    )
    return StrategySpec(
        compression=source.compression,
        collective=source.collective,
        topology=source.topology,
        group_size=source.group_size,
        accumulation_dtype=source.accumulation_dtype,
        error_feedback=source.error_feedback,
        parameter_error_feedback=source.parameter_error_feedback,
        overlap=source.overlap,
        workspace_budget_bytes=source.workspace_budget_bytes,
    )


def _validate_capability_fields(capability: BackendCapability) -> None:
    """Reject non-deterministic values from a capability key."""
    if type(capability.backend_id) is not str:
        raise CompileError("Backend identifier must be a string.")
    if type(capability.strategy) is not StrategySpec:
        raise CompileError("Capability strategy must be a StrategySpec.")
    StrategySpec.__post_init__(capability.strategy)
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
