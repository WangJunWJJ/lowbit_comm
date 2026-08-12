"""Compile-time registry of complete backend targets."""

from __future__ import annotations

from lowbit_comm.core.backend import CommunicationBackend


class BackendRegistry:
    """Maps one target name to one backend implementation."""

    def __init__(self) -> None:
        self._backends: dict[str, CommunicationBackend] = {}

    def register(self, target: str, backend: CommunicationBackend) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("backend target must be a non-empty string")
        if target in self._backends:
            raise KeyError(f"backend target already registered: {target}")
        if backend.name != target:
            raise ValueError(
                f"backend name {backend.name!r} does not match target {target!r}"
            )
        self._backends[target] = backend

    def resolve(self, target: str) -> CommunicationBackend:
        try:
            return self._backends[target]
        except KeyError as error:
            raise KeyError(f"unknown backend target: {target}") from error
