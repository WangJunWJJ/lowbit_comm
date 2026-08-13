"""Immutable backend-lowered communication program types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .context import CompileContext, RuntimeBindings
from .operations import ReductionContract
from .program import CommunicationProgram


class ExecutorKind(Enum):
    """Backend-resolved executable family selected during lowering."""

    NATIVE_ALL_REDUCE = "native_all_reduce"
    COMPRESSED_ALL_GATHER = "compressed_all_gather"
    REDUCED_SHARD = "reduced_shard"
    COMPRESSED_RS_AG = "compressed_rs_ag"


class StageKind(Enum):
    """Physical category of one executable stage."""

    KERNEL = "kernel"
    COLLECTIVE = "collective"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class LoweredStage:
    """One observable backend operation and its cross-rank wire type."""

    name: str
    wire: object
    collective: bool = False
    stage_id: str = ""
    primitive: str = ""
    kind: StageKind = StageKind.KERNEL
    dependencies: tuple[str, ...] = ()
    stream_role: str = "compute"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("lowered stage name must be a non-empty string")
        if self.wire is None:
            raise TypeError("lowered stage wire must not be None")
        if not isinstance(self.kind, StageKind):
            raise TypeError("lowered stage kind must be a StageKind")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not self.stream_role:
            raise ValueError("lowered stage stream role must be non-empty")


@dataclass(frozen=True, slots=True)
class LoweredProgram:
    """Semantic program plus backend-resolved stages and runtime bindings."""

    target: str
    program: CommunicationProgram
    stages: tuple[LoweredStage, ...]
    reduction: ReductionContract
    context: CompileContext
    bindings: RuntimeBindings
    executor_kind: ExecutorKind

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("lowered target must be a non-empty string")
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("lowered program must contain at least one stage")
        if not isinstance(self.executor_kind, ExecutorKind):
            raise TypeError("executor_kind must be an ExecutorKind")
