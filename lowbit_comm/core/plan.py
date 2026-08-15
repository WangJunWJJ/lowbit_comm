"""Immutable compilation context and execution-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from inspect import getattr_static
from types import FunctionType, MemberDescriptorType, MethodType
from typing import TYPE_CHECKING, Callable, cast

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import StrategySpec
from lowbit_comm.core.environment import EnvironmentFingerprint
from lowbit_comm.core.errors import CompileError

if TYPE_CHECKING:
    from lowbit_comm.backends.protocols import BackendPlan


_MISSING_MEMBER = object()


class PlanOrigin(str, Enum):
    """The policy path that produced an execution plan."""

    NATIVE = "native"
    EXPLICIT = "explicit"
    AUTO = "auto"
    NATIVE_FALLBACK = "native_fallback"


@dataclass(frozen=True, slots=True)
class CompilationContext:
    """Immutable environment and resource inputs used by compilation."""

    environment: EnvironmentFingerprint
    workspace_budget_bytes: int
    node_count: int
    workload_class: str
    bucket_min_bytes: int
    bucket_max_bytes: int

    def __post_init__(self) -> None:
        if type(self.environment) is not EnvironmentFingerprint:
            raise CompileError(
                "Compilation environment must be an "
                "EnvironmentFingerprint."
            )
        if (
            type(self.workspace_budget_bytes) is not int
            or self.workspace_budget_bytes < 0
        ):
            raise CompileError(
                "Compilation workspace budget must be a non-negative "
                "integer."
            )
        if type(self.node_count) is not int or self.node_count <= 0:
            raise CompileError(
                "Compilation node count must be a positive integer."
            )
        if type(self.workload_class) is not str or not self.workload_class:
            raise CompileError(
                "Compilation workload class must be a non-empty string."
            )
        if type(self.bucket_min_bytes) is not int:
            raise CompileError(
                "Compilation bucket minimum must be an integer."
            )
        if type(self.bucket_max_bytes) is not int:
            raise CompileError(
                "Compilation bucket maximum must be an integer."
            )
        if self.bucket_min_bytes < 0 or self.bucket_max_bytes < 0:
            raise CompileError(
                "Compilation bucket bounds must be non-negative."
            )
        if self.bucket_min_bytes > self.bucket_max_bytes:
            raise CompileError(
                "Compilation bucket minimum cannot exceed its maximum."
            )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """A fully resolved backend plan with deterministic provenance."""

    intent: CommunicationIntent
    strategy: StrategySpec
    backend_id: str
    backend_plan: BackendPlan
    origin: PlanOrigin
    signature: str
    evidence_fingerprint: str | None

    def __post_init__(self) -> None:
        if type(self.intent) is not CommunicationIntent:
            raise CompileError("Plan intent must be CommunicationIntent.")
        if type(self.strategy) is not StrategySpec:
            raise CompileError("Plan strategy must be StrategySpec.")
        if type(self.backend_id) is not str or not self.backend_id:
            raise CompileError(
                "Plan backend identifier must be a non-empty string."
            )
        _validate_backend_plan(self.backend_plan)
        if type(self.origin) is not PlanOrigin:
            raise CompileError("Plan origin must be PlanOrigin.")
        if type(self.signature) is not str or not self.signature:
            raise CompileError("Plan signature must be a non-empty string.")
        if self.evidence_fingerprint is not None and type(
            self.evidence_fingerprint
        ) is not str:
            raise CompileError(
                "Plan evidence fingerprint must be a string when set."
            )


def _validate_backend_plan(backend_plan: object) -> None:
    message = "ExecutionPlan backend plan must provide callable execute()."
    _validate_callable_member(backend_plan, "execute", message)


def _validate_callable_member(
    value: object,
    member_name: str,
    message: str,
) -> None:
    """Validate callable structure without evaluating descriptors."""
    _resolve_static_callable_member(value, member_name, message)


def _resolve_static_callable_member(
    value: object,
    member_name: str,
    message: str,
) -> Callable[..., object]:
    """Resolve a callable without dynamic attribute access."""
    try:
        from_instance = False
        member = getattr_static(
            value,
            member_name,
            _MISSING_MEMBER,
        )
        try:
            instance_values = object.__getattribute__(value, "__dict__")
        except AttributeError:
            instance_values = None
        if type(instance_values) is dict:
            from_instance = member_name in instance_values
        if type(member) is MemberDescriptorType:
            member = MemberDescriptorType.__get__(
                member,
                value,
                type(value),
            )
            from_instance = True
        member_type = type(member)
        if member_type is staticmethod:
            member = member.__func__
        elif member_type is classmethod:
            member = MethodType(member.__func__, type(value))
        elif (
            issubclass(member_type, staticmethod)
            or issubclass(member_type, classmethod)
        ):
            raise CompileError(message)
        elif member_type is FunctionType and not from_instance:
            member = MethodType(member, value)
        if (
            member is _MISSING_MEMBER
            or isinstance(member, property)
            or not callable(member)
        ):
            raise CompileError(message)
        return cast(Callable[..., object], member)
    except CompileError:
        raise
    except Exception as error:
        raise CompileError(message) from error
