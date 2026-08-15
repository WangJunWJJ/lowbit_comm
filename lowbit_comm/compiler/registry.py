"""Deterministic registry of backend capabilities."""

from __future__ import annotations

from collections.abc import Iterable

from lowbit_comm.api.intent import CommunicationIntent
from lowbit_comm.api.policy import StrategySpec
from lowbit_comm.backends.protocols import Backend, BackendCapability
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
)
from lowbit_comm.core.plan import (
    _resolve_static_callable_member,
    _resolve_static_member,
)
from lowbit_comm.core.signatures import StrategyKey, strategy_key

BackendMatch = tuple[BackendCapability, Backend]
CapabilityKey = tuple[
    str,
    StrategyKey,
    str,
    int,
    bool,
    int,
    tuple[str, ...],
    bool,
]


class BackendRegistry:
    """Bind advertised backend capabilities to their declaring backend."""

    def __init__(self, backends: Iterable[Backend] = ()) -> None:
        self._entries: dict[CapabilityKey, BackendMatch] = {}
        self._generation = 0
        for backend in backends:
            self.register(backend)

    @property
    def generation(self) -> int:
        """Return the monotonic generation of registered capabilities."""
        return self._generation

    def register(self, backend: Backend) -> None:
        """Register all immutable capabilities declared by *backend*."""
        backend_id = _resolve_backend_id(backend)
        capabilities_method = _resolve_static_callable_member(
            backend,
            "capabilities",
            "Backend must provide callable capabilities().",
        )
        _resolve_static_callable_member(
            backend,
            "lower",
            "Backend must provide callable lower().",
        )
        try:
            capabilities = capabilities_method()
        except CompileError:
            raise
        except Exception as error:
            raise CompileError(
                "Backend capabilities() failed during registration."
            ) from error
        if type(capabilities) is not tuple or not capabilities:
            raise CompileError(
                "Backend capabilities must be a non-empty tuple."
            )
        entries: dict[CapabilityKey, BackendMatch] = {}
        for capability in capabilities:
            if type(capability) is not BackendCapability:
                raise CompileError(
                    "Backend capability must be BackendCapability."
                )
            capability.__post_init__()
            if capability.backend_id != backend_id:
                raise CompileError(
                    "Capability identifier must match its backend."
                )
            key = _capability_key(capability)
            if key in self._entries or key in entries:
                raise CapabilityError("Duplicate backend capability key.")
            entries[key] = (capability, backend)
        self._entries.update(entries)
        if entries:
            self._generation += 1

    def resolve_exact(self, capability: BackendCapability) -> BackendMatch:
        """Return the backend bound to one exact capability declaration."""
        if type(capability) is not BackendCapability:
            raise CompileError("Capability lookup requires BackendCapability.")
        try:
            return self._entries[_capability_key(capability)]
        except KeyError as error:
            raise CapabilityError(
                "Backend capability is unavailable."
            ) from error

    def candidates(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> tuple[BackendMatch, ...]:
        """Return key-sorted capabilities matching an exact compile request."""
        if type(intent) is not CommunicationIntent:
            raise CompileError(
                "Candidate lookup requires CommunicationIntent."
            )
        if type(strategy) is not StrategySpec:
            raise CompileError(
                "Candidate lookup requires StrategySpec."
            )
        return tuple(
            match for _, match in sorted(self._entries.items())
            if match[0].supports(intent, strategy)
        )

    def capabilities_for_world_size(
        self,
        world_size: int,
    ) -> tuple[BackendMatch, ...]:
        """Return diagnostic-only capability matches for *world_size*."""
        if type(world_size) is not int or world_size <= 0:
            raise CompileError(
                "Candidate world size must be a positive integer."
            )
        return tuple(
            match for _, match in sorted(self._entries.items())
            if _supports_world_size(match[0], world_size)
        )


def _capability_key(capability: BackendCapability) -> CapabilityKey:
    """Return the complete, sortable key for a capability declaration."""
    return (
        capability.backend_id,
        strategy_key(capability.strategy),
        capability.output.value,
        capability.min_world_size,
        capability.max_world_size is None,
        capability.max_world_size or 0,
        tuple(sorted(capability.supported_dtypes)),
        capability.supports_async,
    )


def _resolve_backend_id(backend: object) -> str:
    """Read one stable identifier without evaluating user descriptors."""
    backend_id, _ = _resolve_static_member(
        backend,
        "backend_id",
        "Backend identifier must be a non-empty string.",
    )
    if type(backend_id) is not str or not backend_id:
        raise CompileError(
            "Backend identifier must be a non-empty string."
        )
    return backend_id


def _supports_world_size(
    capability: BackendCapability,
    world_size: int,
) -> bool:
    """Return whether a capability includes *world_size*."""
    return (
        capability.min_world_size <= world_size
        and (capability.max_world_size is None
             or world_size <= capability.max_world_size)
    )
