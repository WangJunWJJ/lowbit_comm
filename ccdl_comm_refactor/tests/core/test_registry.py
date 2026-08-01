from __future__ import annotations

import pytest

from ccdl_comm.backend import BackendCapabilities
from ccdl_comm.exceptions import BackendRegistrationError, UnsupportedCollective
from ccdl_comm.registry import BackendKey, BackendRegistry


class FakeExecutor:
    execution_info = object()

    def run(self, tensor: object) -> object:
        return tensor


class FakeBackend:
    name = "cuda"
    abi_version = 1

    def capabilities(self, context: object) -> BackendCapabilities:
        return BackendCapabilities(backend=self.name, available=True)

    def compile(self, plan: object, context: object) -> FakeExecutor:
        return FakeExecutor()


def test_registry_rejects_duplicate_key() -> None:
    registry = BackendRegistry()
    key = BackendKey("all_reduce", "ring", "cuda", "full")
    registry.register(key, FakeBackend)

    with pytest.raises(BackendRegistrationError, match="already registered"):
        registry.register(key, FakeBackend)


def test_registry_missing_key_is_diagnostic() -> None:
    registry = BackendRegistry()

    with pytest.raises(UnsupportedCollective, match="all_reduce:ring:cuda:full"):
        registry.resolve(BackendKey("all_reduce", "ring", "cuda", "full"))


def test_registry_resolves_fresh_backend_from_factory() -> None:
    registry = BackendRegistry()
    key = BackendKey("all_reduce", "ring", "cuda", "full")
    registry.register(key, FakeBackend)

    first = registry.resolve(key)
    second = registry.resolve(key)

    assert isinstance(first, FakeBackend)
    assert isinstance(second, FakeBackend)
    assert first is not second
    assert key in registry
    assert registry.keys() == (key,)


def test_registry_rejects_factory_that_returns_non_backend() -> None:
    registry = BackendRegistry()
    key = BackendKey("all_reduce", "ring", "cuda", "full")
    registry.register(key, lambda: object())

    with pytest.raises(BackendRegistrationError, match="CommunicationBackend"):
        registry.resolve(key)


@pytest.mark.parametrize("field", ["collective", "strategy", "backend", "output_layout"])
def test_backend_key_rejects_empty_fields(field: str) -> None:
    values = {
        "collective": "all_reduce",
        "strategy": "ring",
        "backend": "cuda",
        "output_layout": "full",
    }
    values[field] = " "

    with pytest.raises(ValueError, match=field):
        BackendKey(**values)
