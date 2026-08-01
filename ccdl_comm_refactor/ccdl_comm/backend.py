"""Pure Core capability data and communication backend protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .executor import CompiledExecutor
from .plan import CommunicationPlan, CompileContext
from .stage import _require_non_empty


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable facts reported by one backend for a compile context."""

    backend: str
    available: bool
    collectives: frozenset[str] = frozenset()
    strategies: frozenset[str] = frozenset()
    dtypes: frozenset[str] = frozenset()
    bits: frozenset[int] = frozenset()
    output_layouts: frozenset[str] = frozenset()
    supports_async: bool = False
    supports_dynamic_shape: bool = False
    features: frozenset[str] = frozenset()
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.backend, "backend")
        for field_name in ("collectives", "strategies", "dtypes", "output_layouts", "features"):
            values = frozenset(getattr(self, field_name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, values)

        bits = frozenset(self.bits)
        if any(isinstance(bit, bool) or not isinstance(bit, int) or bit <= 0 for bit in bits):
            raise ValueError("bits must contain positive integers")
        object.__setattr__(self, "bits", bits)

        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning.strip() for warning in warnings):
            raise ValueError("warnings must contain non-empty strings")
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

        if not self.available and (self.reason is None or not self.reason.strip()):
            raise ValueError("reason is required when backend is unavailable")


@runtime_checkable
class CommunicationBackend(Protocol):
    """Control-plane interface implemented by concrete communication backends."""

    name: str
    abi_version: int

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        """Return immutable capabilities for a static compile context."""

        ...

    def compile(self, plan: CommunicationPlan, context: CompileContext) -> CompiledExecutor:
        """Bind a validated plan to a reusable data-path executor."""

        ...
