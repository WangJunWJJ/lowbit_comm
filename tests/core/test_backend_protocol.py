from __future__ import annotations

import pytest

from ccdl_comm.backend import BackendCapabilities, CommunicationBackend
from ccdl_comm.executor import CompiledExecutor


def test_backend_capabilities_are_immutable_and_own_collections() -> None:
    collectives = {"all_reduce"}
    details = {"arch": "sm_86"}
    capabilities = BackendCapabilities(
        backend="cuda",
        available=True,
        collectives=collectives,
        strategies={"all_gather"},
        dtypes={"fp16"},
        bits={8},
        output_layouts={"full"},
        details=details,
    )
    collectives.clear()
    details.clear()

    assert capabilities.collectives == frozenset({"all_reduce"})
    assert capabilities.details == {"arch": "sm_86"}
    with pytest.raises(TypeError):
        capabilities.details["arch"] = "sm_89"  # type: ignore[index]


def test_backend_capabilities_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="backend"):
        BackendCapabilities(backend="", available=True)
    with pytest.raises(ValueError, match="bits"):
        BackendCapabilities(backend="cuda", available=True, bits={0})
    with pytest.raises(ValueError, match="reason"):
        BackendCapabilities(backend="cuda", available=False)
    with pytest.raises(ValueError, match="verified_strategies"):
        BackendCapabilities(
            backend="cuda",
            available=True,
            strategies={"all_gather"},
            verified_strategies={"topology"},
        )


def test_backend_and_executor_protocols_are_runtime_checkable() -> None:
    assert getattr(CommunicationBackend, "_is_runtime_protocol", False) is True
    assert getattr(CompiledExecutor, "_is_runtime_protocol", False) is True
