"""Immutable record of a compiler strategy decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    requested_algorithm: str
    effective_algorithm: str
    requested_wire: object
    effective_wire: object
    physical_primitive: str
    requested_output: str = "unknown"
    effective_output: str = "unknown"
    logical_bytes: int = 0
    estimated_wire_bytes: int = 0
    fused_stages: tuple[str, ...] = ()
    workspace_bytes: int = 0
    topology_signature: str = "unknown"
    world_size: int = 1
    fallback_reason: str | None = None
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fused_stages", tuple(self.fused_stages))
        for name in ("logical_bytes", "estimated_wire_bytes", "workspace_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
