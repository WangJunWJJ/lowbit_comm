"""Deterministic registry of backend capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import NamedTuple, cast

from lowbit_comm.api.intent import (
    CommunicationIntent,
    _validate_communication_intent_graph,
)
from lowbit_comm.api.policy import (
    StrategySpec,
    _validate_strategy_graph,
)
from lowbit_comm.backends.protocols import (
    Backend,
    BackendCapability,
    BackendPlan,
    _snapshot_backend_capability,
    _supports_request,
)
from lowbit_comm.core.errors import (
    CapabilityError,
    CompileError,
)
from lowbit_comm.core.plan import (
    _resolve_static_callable_member,
    _resolve_static_member,
    _same_bound_callable,
)
from lowbit_comm.core.signatures import StrategyKey, strategy_key

BackendMatch = tuple[BackendCapability, Backend]
_LowerCallable = Callable[
    [CommunicationIntent, StrategySpec],
    BackendPlan,
]
_LoweringMatch = tuple[BackendCapability, _LowerCallable]
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


class _BackendEntry(NamedTuple):
    """Tuple-backed immutable binding retained only inside Registry."""

    capability: BackendCapability
    backend: Backend
    lower: _LowerCallable


class BackendRegistry:
    """Bind advertised backend capabilities to their declaring backend."""

    def __init__(self, backends: Iterable[Backend] = ()) -> None:
        self._entries: dict[CapabilityKey, _BackendEntry] = {}
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
        committed_items = _validated_entry_items(self._entries)
        _require_backend_identity_owner(
            (entry for _, entry in committed_items),
            backend_id,
            backend,
        )
        capabilities_method = _resolve_static_callable_member(
            backend,
            "capabilities",
            "Backend must provide callable capabilities().",
        )
        lower = cast(
            _LowerCallable,
            _resolve_static_callable_member(
                backend,
                "lower",
                "Backend must provide callable lower().",
            ),
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
            raise CompileError("Backend capabilities must be a non-empty tuple.")
        entries: dict[CapabilityKey, _BackendEntry] = {}
        for capability in capabilities:
            snapshot = _snapshot_backend_capability(capability)
            if snapshot.backend_id != backend_id:
                raise CompileError("Capability identifier must match its backend.")
            key = _capability_key(snapshot)
            if key in self._entries or key in entries:
                raise CapabilityError("Duplicate backend capability key.")
            entries[key] = _BackendEntry(snapshot, backend, lower)
        self._entries.update(entries)
        if entries:
            self._generation += 1

    def resolve_exact(self, capability: BackendCapability) -> BackendMatch:
        """Return the backend bound to one exact capability declaration."""
        lookup = _snapshot_backend_capability(capability)
        _validated_entry_items(self._entries)
        try:
            return _backend_match(self._entries[_capability_key(lookup)])
        except KeyError as error:
            raise CapabilityError("Backend capability is unavailable.") from error

    def _resolve_lowering(
        self,
        capability: BackendCapability,
    ) -> _LoweringMatch:
        """Return the registered callable for one exact capability."""
        lookup = _snapshot_backend_capability(capability)
        _validated_entry_items(self._entries)
        try:
            entry = self._entries[_capability_key(lookup)]
        except KeyError as error:
            raise CapabilityError(
                "Backend capability is unavailable for lowering."
            ) from error
        return _snapshot_backend_capability(entry.capability), entry.lower

    def candidates(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> tuple[BackendMatch, ...]:
        """Return key-sorted capabilities matching an exact compile request."""
        request = _validate_communication_intent_graph(intent)
        requested_strategy = _validate_strategy_graph(strategy)
        entries = _validated_entry_items(self._entries)
        return tuple(
            _backend_match(entry)
            for _, entry in sorted(entries)
            if _supports_request(
                entry.capability,
                request,
                requested_strategy,
            )
        )

    def capabilities_for_world_size(
        self,
        world_size: int,
    ) -> tuple[BackendMatch, ...]:
        """Return diagnostic-only capability matches for *world_size*."""
        if type(world_size) is not int or world_size <= 0:
            raise CompileError("Candidate world size must be a positive integer.")
        entries = _validated_entry_items(self._entries)
        return tuple(
            _backend_match(entry)
            for _, entry in sorted(entries)
            if _supports_world_size(entry.capability, world_size)
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


def _backend_match(entry: _BackendEntry) -> BackendMatch:
    """Return the stable public/diagnostic two-tuple for one entry."""
    return _snapshot_backend_capability(entry.capability), entry.backend


def _validated_entry_items(
    entries: dict[CapabilityKey, _BackendEntry],
) -> tuple[tuple[CapabilityKey, _BackendEntry], ...]:
    """Validate that every internal key owns its trusted snapshot value."""
    validated: list[tuple[CapabilityKey, _BackendEntry]] = []
    for key, entry in entries.items():
        validated_entry, snapshot = _validate_backend_entry(entry)
        if _capability_key(snapshot) != key:
            raise CompileError("Registry internal capability key is invalid.")
        validated.append((key, validated_entry))
    return tuple(validated)


def _validate_backend_entry(
    entry: object,
) -> tuple[_BackendEntry, BackendCapability]:
    """Validate one exact tuple entry and its saved lowering binding."""
    message = "Registry internal capability entry is invalid."
    if type(entry) is not _BackendEntry or len(entry) != 3:
        raise CompileError(message)
    exact_entry = cast(_BackendEntry, entry)
    try:
        snapshot = _snapshot_backend_capability(exact_entry.capability)
        backend_id = _resolve_backend_id(exact_entry.backend)
        resolved_lower = _resolve_static_callable_member(
            exact_entry.backend,
            "lower",
            message,
        )
    except CompileError as error:
        raise CompileError(message) from error
    if snapshot.backend_id != backend_id or not _same_bound_callable(
        exact_entry.lower,
        resolved_lower,
    ):
        raise CompileError(message)
    return exact_entry, snapshot


def _require_backend_identity_owner(
    entries: Iterable[_BackendEntry],
    backend_id: str,
    backend: object,
) -> None:
    """Require one committed object identity to own *backend_id*."""
    owner: object | None = None
    for entry in entries:
        if entry.capability.backend_id != backend_id:
            continue
        if owner is None:
            owner = entry.backend
        elif entry.backend is not owner:
            raise CapabilityError(
                "Registry backend identity ownership is inconsistent."
            )
    if owner is not None and backend is not owner:
        raise CapabilityError(
            "Backend identifier is already owned by another backend object."
        )


def _resolve_backend_id(backend: object) -> str:
    """Read one stable identifier without evaluating user descriptors."""
    backend_id, _ = _resolve_static_member(
        backend,
        "backend_id",
        "Backend identifier must be a non-empty string.",
    )
    if type(backend_id) is not str or not backend_id:
        raise CompileError("Backend identifier must be a non-empty string.")
    return backend_id


def _supports_world_size(
    capability: BackendCapability,
    world_size: int,
) -> bool:
    """Return whether a capability includes *world_size*."""
    return capability.min_world_size <= world_size and (
        capability.max_world_size is None or world_size <= capability.max_world_size
    )
