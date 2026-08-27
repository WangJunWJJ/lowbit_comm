"""Runtime compatibility probes for experimental RSAG/qWD."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from lowbit_comm import CapabilityError
from lowbit_comm.backends.cuda.loader import CUDA_ABI_VERSION
from lowbit_comm.experimental import compatibility
from lowbit_comm.experimental.compatibility import (
    RSAG_CUDA_EXTENSION_ABI,
    RSAGRuntimeABI,
    RSAG_VERIFIED_RUNTIME_MATRIX,
    is_verified_rsag_runtime,
    probe_rsag_compatibility,
)


def _runtime(**changes: object) -> RSAGRuntimeABI:
    values: dict[str, object] = {
        "torch_version": "2.5.0a0+872d972e41.nv24.08",
        "cuda_version": "12.6",
        "nccl_version": "2.22.3",
        "cuda_extension_abi": 1,
    }
    values.update(changes)
    return RSAGRuntimeABI(**values)  # type: ignore[arg-type]


def test_verified_matrix_is_exact_and_matches_the_extension_loader() -> None:
    assert RSAG_CUDA_EXTENSION_ABI == CUDA_ABI_VERSION
    assert RSAG_VERIFIED_RUNTIME_MATRIX == (_runtime(),)
    assert is_verified_rsag_runtime(_runtime()) is True


def test_build_fingerprint_is_deterministic_and_content_sensitive(
    monkeypatch,
) -> None:
    compute = getattr(compatibility, "compute_rsag_build_fingerprint", None)
    assert callable(compute)
    values = {
        name: f"content:{name}".encode()
        for name in compatibility._RSAG_FINGERPRINT_MODULES
    }
    monkeypatch.setattr(
        compatibility,
        "_read_module_bytes",
        lambda name: values[name],
    )

    first = compute()
    second = compute()
    values[compatibility._RSAG_FINGERPRINT_MODULES[0]] += b"-changed"
    changed = compute()

    assert len(first) == 64
    assert first == second
    assert changed != first


def test_build_fingerprint_fails_closed_when_any_module_is_unreadable(
    monkeypatch,
) -> None:
    compute = getattr(compatibility, "compute_rsag_build_fingerprint", None)
    assert callable(compute)
    def unreadable(name: str) -> bytes:
        raise OSError(name)

    monkeypatch.setattr(compatibility, "_read_module_bytes", unreadable)

    with pytest.raises(CapabilityError, match="build fingerprint"):
        compute()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch_version", "2.6.0"),
        ("cuda_version", "12.8"),
        ("nccl_version", "2.23.0"),
        ("cuda_extension_abi", 2),
    ],
)
def test_any_unverified_runtime_field_is_rejected(
    field: str,
    value: object,
) -> None:
    runtime = replace(_runtime(), **{field: value})

    assert is_verified_rsag_runtime(runtime) is False


def test_runtime_identity_requires_exact_builtin_fields() -> None:
    with pytest.raises(ValueError, match="cuda_extension_abi"):
        _runtime(cuda_extension_abi=True)
    with pytest.raises(ValueError, match="torch_version"):
        _runtime(torch_version=" unknown ")


def test_probe_reports_a_verified_live_runtime(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            nccl=SimpleNamespace(version=lambda: (2, 22, 3)),
        ),
    )
    fake_extension = SimpleNamespace(abi_version=lambda: 1)
    fake_loader = SimpleNamespace(load_extension=lambda: fake_extension)

    def fake_import(name: str) -> object:
        return {
            "torch": fake_torch,
            "lowbit_comm.backends.cuda.loader": fake_loader,
        }[name]

    monkeypatch.setattr(compatibility, "import_module", fake_import)

    report = probe_rsag_compatibility()

    assert report.runtime == _runtime()
    assert report.extension_loaded is True
    assert report.compatible is True
    assert report.reason == "verified_runtime"


def test_probe_reports_missing_cuda_without_loading_the_extension(
    monkeypatch,
) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.5.0a0+872d972e41.nv24.08",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    loaded = False

    def fake_import(name: str) -> object:
        nonlocal loaded
        if name == "torch":
            return fake_torch
        loaded = True
        raise AssertionError("extension loader must not be imported")

    monkeypatch.setattr(compatibility, "import_module", fake_import)

    report = probe_rsag_compatibility()

    assert report.runtime is None
    assert report.extension_loaded is False
    assert report.compatible is False
    assert report.reason == "cuda_unavailable"
    assert loaded is False
