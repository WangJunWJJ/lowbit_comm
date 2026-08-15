"""Immutable strategy and policy contracts."""

from dataclasses import dataclass
from enum import Enum

from lowbit_comm.core.errors import CompileError


class CompressionKind(str, Enum):
    """Wire compression choices available to a strategy."""

    NONE = "none"
    INT8 = "int8"


class CollectiveKind(str, Enum):
    """Collective algorithm families available to a strategy."""

    NATIVE = "native"
    COMPRESSED_ALL_GATHER_REDUCE = "compressed_all_gather_reduce"


class TopologyKind(str, Enum):
    """Topology choices available to a strategy."""

    BACKEND_DEFAULT = "backend_default"
    RING = "ring"
    TREE = "tree"


class AccumulationDType(str, Enum):
    """Dtypes used while accumulating a communication result."""

    FP16 = "fp16"
    FP32 = "fp32"


def _require_enum(value: object, enum_type: type[Enum], name: str) -> None:
    """Raise when *value* is not a member of *enum_type*."""
    if not isinstance(value, enum_type):
        raise CompileError(f"Strategy {name} must be a {enum_type.__name__}.")


def _validate_group_size(
    compression: CompressionKind,
    group_size: int | None,
) -> None:
    """Validate quantized-group configuration."""
    if group_size is not None and (
        type(group_size) is not int
        or group_size <= 0
    ):
        raise CompileError("Group size must be a positive integer when set.")
    if compression is CompressionKind.INT8 and group_size is None:
        raise CompileError("INT8 compression requires a positive group size.")
    if compression is CompressionKind.NONE and group_size is not None:
        raise CompileError("NONE compression does not accept a group size.")


def _validate_workspace_budget(workspace_budget_bytes: int | None) -> None:
    """Validate an optional non-negative workspace budget."""
    if workspace_budget_bytes is not None and (
        type(workspace_budget_bytes) is not int
        or workspace_budget_bytes < 0
    ):
        raise CompileError("Workspace budget must be a non-negative integer.")


def _validate_bool(value: object, name: str) -> None:
    """Validate that a strategy flag is exactly a boolean."""
    if type(value) is not bool:
        raise CompileError(f"Strategy {name} must be a boolean.")


def _is_supported_combination(
    compression: CompressionKind,
    collective: CollectiveKind,
) -> bool:
    """Return whether compression and collective belong to one family."""
    if compression is CompressionKind.NONE:
        return collective is CollectiveKind.NATIVE
    return collective is CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE


def _validate_enum_set(
    values: frozenset[Enum] | None,
    enum_type: type[Enum],
    name: str,
    *,
    allow_none: bool = True,
) -> None:
    """Validate that an optional enum restriction is an immutable set."""
    if values is None:
        if allow_none:
            return
        raise CompileError(f"{name.capitalize()} must be a frozenset.")
    if type(values) is not frozenset:
        raise CompileError(f"{name.capitalize()} must be a frozenset.")
    if not all(isinstance(value, enum_type) for value in values):
        raise CompileError(f"{name.capitalize()} contain invalid enum values.")


def _reject_conflicting_constraints(
    allowed: frozenset[Enum] | None,
    denied: frozenset[Enum],
    kind: str,
) -> None:
    """Reject a constraint that both allows and denies the same value."""
    if allowed is not None and allowed & denied:
        raise CompileError(f"{kind.capitalize()} constraints cannot conflict.")


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """An explicit, backend-independent communication strategy."""

    compression: CompressionKind
    collective: CollectiveKind
    topology: TopologyKind
    group_size: int | None = None
    accumulation_dtype: AccumulationDType = AccumulationDType.FP32
    error_feedback: bool = False
    parameter_error_feedback: bool = False
    overlap: bool = False
    workspace_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        _require_enum(self.compression, CompressionKind, "compression")
        _require_enum(self.collective, CollectiveKind, "collective")
        _require_enum(self.topology, TopologyKind, "topology")
        _require_enum(
            self.accumulation_dtype,
            AccumulationDType,
            "accumulation dtype",
        )
        _validate_bool(self.error_feedback, "error feedback")
        _validate_bool(
            self.parameter_error_feedback,
            "parameter error feedback",
        )
        _validate_bool(self.overlap, "overlap")
        _validate_group_size(self.compression, self.group_size)
        _validate_workspace_budget(self.workspace_budget_bytes)
        if not _is_supported_combination(self.compression, self.collective):
            raise CompileError(
                "Compression and collective kinds are contradictory."
            )


@dataclass(frozen=True, slots=True)
class AutoConstraints:
    """Immutable restrictions for compiler-selected strategies."""

    allowed_compressions: frozenset[CompressionKind] | None = None
    denied_compressions: frozenset[CompressionKind] = frozenset()
    allowed_collectives: frozenset[CollectiveKind] | None = None
    denied_collectives: frozenset[CollectiveKind] = frozenset()
    allowed_topologies: frozenset[TopologyKind] | None = None
    denied_topologies: frozenset[TopologyKind] = frozenset()
    max_workspace_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_enum_set(
            self.allowed_compressions,
            CompressionKind,
            "allowed compressions",
        )
        _validate_enum_set(
            self.denied_compressions,
            CompressionKind,
            "denied compressions",
            allow_none=False,
        )
        _validate_enum_set(
            self.allowed_collectives,
            CollectiveKind,
            "allowed collectives",
        )
        _validate_enum_set(
            self.denied_collectives,
            CollectiveKind,
            "denied collectives",
            allow_none=False,
        )
        _validate_enum_set(
            self.allowed_topologies,
            TopologyKind,
            "allowed topologies",
        )
        _validate_enum_set(
            self.denied_topologies,
            TopologyKind,
            "denied topologies",
            allow_none=False,
        )
        _validate_workspace_budget(self.max_workspace_bytes)
        _reject_conflicting_constraints(
            self.allowed_compressions,
            self.denied_compressions,
            "compression",
        )
        _reject_conflicting_constraints(
            self.allowed_collectives,
            self.denied_collectives,
            "collective",
        )
        _reject_conflicting_constraints(
            self.allowed_topologies,
            self.denied_topologies,
            "topology",
        )


@dataclass(frozen=True, slots=True)
class NativePolicy:
    """Require the compiler to use the native collective path."""


@dataclass(frozen=True, slots=True)
class AutoPolicy:
    """Allow the compiler to select a strategy within constraints."""

    constraints: AutoConstraints = AutoConstraints()

    def __post_init__(self) -> None:
        if type(self.constraints) is not AutoConstraints:
            raise CompileError(
                "Auto-policy constraints must be AutoConstraints."
            )


@dataclass(frozen=True, slots=True)
class ExplicitPolicy:
    """Require the compiler to use exactly one strategy specification."""

    strategy: StrategySpec

    def __post_init__(self) -> None:
        if type(self.strategy) is not StrategySpec:
            raise CompileError(
                "Explicit-policy strategy must be a StrategySpec."
            )
