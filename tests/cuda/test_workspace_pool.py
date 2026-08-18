"""CUDA workspace lease lifecycle tests."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path

import pytest

from lowbit_comm import ExecutionError


ROOT = Path(__file__).resolve().parents[2]


def test_workspace_lease_exposes_read_only_storage_to_native_plans() -> None:
    header = (ROOT / "csrc" / "runtime" / "workspace_pool.h").read_text(
        encoding="utf-8"
    )

    assert "const torch::Tensor& storage() const noexcept;" in header


@pytest.fixture()
def fake_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for workspace allocation")
    extension = importlib.import_module("lowbit_comm._C")
    extension.reset_test_workspace_pool(capacity_bytes=1024)
    return extension


def test_workspace_pool_never_reuses_inflight_lease(fake_extension) -> None:
    first = fake_extension.acquire_test_lease(1024)
    with pytest.raises(ExecutionError, match="workspace pool"):
        fake_extension.acquire_test_lease(1024)

    first.complete_for_test()
    second = fake_extension.acquire_test_lease(1024)

    assert second.lease_id != first.lease_id


def test_workspace_pool_rejects_request_over_capacity(fake_extension) -> None:
    with pytest.raises(ExecutionError, match="workspace pool"):
        fake_extension.acquire_test_lease(1025)


def test_cuda_work_retains_lease_until_terminal_wait(fake_extension) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    first = executor.run("first", workspace_bytes=1024)

    with pytest.raises(ExecutionError, match="workspace pool"):
        executor.run("blocked", workspace_bytes=1024)

    assert first.wait() == "first"
    second = executor.run("second", workspace_bytes=1024)
    assert second.wait() == "second"


def test_destroyed_work_cleans_event_before_returning_lease(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    work = executor.run("discarded", workspace_bytes=1024)

    del work
    gc.collect()

    replacement = executor.run("replacement", workspace_bytes=1024)
    assert replacement.wait() == "replacement"
