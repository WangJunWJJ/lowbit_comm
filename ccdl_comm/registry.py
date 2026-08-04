"""Control-plane registry for communication backend factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .backend import AutoStrategySelector, CommunicationBackend
from .exceptions import BackendRegistrationError, UnsupportedCollective
from .stage import _require_non_empty


@dataclass(frozen=True)
class BackendKey:
    """Four-dimensional lookup key for a compiled backend implementation."""

    collective: str
    strategy: str
    backend: str
    output_layout: str

    def __post_init__(self) -> None:
        for field_name in ("collective", "strategy", "backend", "output_layout"):
            _require_non_empty(getattr(self, field_name), field_name)

    def __str__(self) -> str:
        return f"{self.collective}:{self.strategy}:{self.backend}:{self.output_layout}"


class BackendRegistry:
    """Register and resolve backend factories outside the execution hot path."""

    def __init__(self) -> None:
        self._factories: dict[BackendKey, Callable[[], CommunicationBackend]] = {}
        self._strategy_selectors: dict[str, AutoStrategySelector] = {}

    def register(self, key: BackendKey, factory: Callable[[], CommunicationBackend]) -> None:
        if not isinstance(key, BackendKey):
            raise TypeError("key must be a BackendKey")
        if not callable(factory):
            raise TypeError("factory must be callable")
        if key in self._factories:
            raise BackendRegistrationError(f"backend key already registered: {key}")
        self._factories[key] = factory

    def resolve(self, key: BackendKey) -> CommunicationBackend:
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise UnsupportedCollective(
                str(key),
                reason="no backend factory is registered for the requested key",
            ) from exc
        backend = factory()
        if not isinstance(backend, CommunicationBackend):
            raise BackendRegistrationError(f"factory for {key} did not return a CommunicationBackend")
        return backend

    def register_strategy_selector(
        self,
        backend: str,
        selector: AutoStrategySelector,
    ) -> None:
        """Register one backend-owned selector outside the execution hot path."""

        _require_non_empty(backend, "backend")
        if not callable(selector):
            raise TypeError("selector must be callable")
        if backend in self._strategy_selectors:
            raise BackendRegistrationError(
                f"strategy selector already registered for backend {backend!r}"
            )
        self._strategy_selectors[backend] = selector

    def strategy_selector(self, backend: str) -> AutoStrategySelector | None:
        """Return a backend selector, or ``None`` for generic Core ordering."""

        _require_non_empty(backend, "backend")
        return self._strategy_selectors.get(backend)

    def __contains__(self, key: object) -> bool:
        return key in self._factories

    def keys(self) -> tuple[BackendKey, ...]:
        return tuple(self._factories)
