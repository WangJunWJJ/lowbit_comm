"""Immutable backend-lowered communication program types."""

from __future__ import annotations

from dataclasses import dataclass

from .context import CompileContext, RuntimeBindings
from .program import CommunicationProgram


@dataclass(frozen=True, slots=True)
class LoweredStage:
    """One observable backend operation and its cross-rank wire type."""

    name: str
    wire: object

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("lowered stage name must be a non-empty string")
        if self.wire is None:
            raise TypeError("lowered stage wire must not be None")


@dataclass(frozen=True, slots=True)
class LoweredProgram:
    """Semantic program plus backend-resolved stages and runtime bindings."""

    target: str
    program: CommunicationProgram
    stages: tuple[LoweredStage, ...]
    context: CompileContext
    bindings: RuntimeBindings

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("lowered target must be a non-empty string")
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("lowered program must contain at least one stage")
