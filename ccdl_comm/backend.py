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
    verified_strategies: frozenset[str] = frozenset()
    async_strategies: frozenset[str] = frozenset()
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
        for field_name in (
            "collectives",
            "strategies",
            "verified_strategies",
            "async_strategies",
            "dtypes",
            "output_layouts",
            "features",
        ):
            values = frozenset(getattr(self, field_name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, values)

        if not self.verified_strategies <= self.strategies:
            raise ValueError("verified_strategies must be a subset of strategies")
        if not self.async_strategies <= self.strategies:
            raise ValueError("async_strategies must be a subset of strategies")
        if self.async_strategies and not self.supports_async:
            raise ValueError("async_strategies require supports_async=True")

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


@dataclass(frozen=True, slots=True)
class StrategyChoice:
    """One backend-provided, explainable compile-time strategy decision."""

    strategy: str
    reason: str
    policy_id: str
    benchmark_matched: bool
    expected_speedup: float | None = None
    observed_speedup: float | None = None
    baseline: str | None = None
    evidence: str | None = None
    fallback: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("strategy", "reason", "policy_id"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.expected_speedup is not None and self.expected_speedup <= 0:
            raise ValueError("expected_speedup must be > 0")
        if self.observed_speedup is not None and self.observed_speedup <= 0:
            raise ValueError("observed_speedup must be > 0")
        if self.observed_speedup is not None and not self.baseline:
            raise ValueError("observed_speedup requires an explicit baseline")
        if self.benchmark_matched and not self.evidence:
            raise ValueError("benchmark-matched choices require evidence")
        fallback = tuple(self.fallback)
        if any(not value.strip() for value in fallback):
            raise ValueError("fallback strategies must not be empty")
        object.__setattr__(self, "fallback", fallback)


@runtime_checkable
class AutoStrategySelector(Protocol):
    """Backend-owned policy injected into the Core compile control plane."""

    def __call__(
        self,
        plan: CommunicationPlan,
        context: CompileContext,
    ) -> StrategyChoice:
        """Select one concrete strategy without executing communication."""

        ...


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
