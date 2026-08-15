"""Immutable compilation context and execution-plan contracts."""

from dataclasses import dataclass
from enum import Enum

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import StrategySpec
from lowbit_comm.backends.protocols import BackendPlan
from lowbit_comm.compiler.evidence import EnvironmentFingerprint
from lowbit_comm.core.errors import CompileError


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
