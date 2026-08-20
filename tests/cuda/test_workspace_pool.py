"""CUDA workspace lease lifecycle tests."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
import subprocess
import sys

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


def test_quarantined_workspace_permanently_rejects_reuse(
    fake_extension,
) -> None:
    lease = fake_extension.acquire_test_lease(1024)

    lease.quarantine_for_test()

    for _ in range(3):
        with pytest.raises(ExecutionError, match="quarantined"):
            fake_extension.acquire_test_lease(1)


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


@pytest.mark.parametrize("mode", ("success", "failure"))
def test_native_cuda_work_waiters_use_isolated_hard_timeout(
    fake_extension,
    mode: str,
) -> None:
    del fake_extension
    worker = Path(__file__).with_name("native_work_concurrency_worker.py")
    process = subprocess.Popen(
        [sys.executable, str(worker), "--mode", mode],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        pytest.fail(
            "native CudaWork waiter subprocess exceeded hard timeout\n"
            f"{output}"
        )
    assert process.returncode == 0, output
    assert f"NATIVE_WORK_CONCURRENCY_OK mode={mode}" in output


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


def test_event_record_failure_quarantines_workspace(fake_extension) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    executor._inject_event_failure_for_test("record")

    with pytest.raises(ExecutionError, match="event record failure"):
        executor.run("must-not-publish", workspace_bytes=1024)

    for _ in range(3):
        with pytest.raises(ExecutionError, match="quarantined"):
            executor.run("must-not-reuse", workspace_bytes=1)


def test_event_sync_failure_is_stable_and_quarantines_workspace(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    executor._inject_event_failure_for_test("synchronize")
    work = executor.run("must-not-publish", workspace_bytes=1024)

    failures = []
    for operation in (work.wait, work.wait, work.result):
        with pytest.raises(
            ExecutionError,
            match="event synchronize failure",
        ) as caught:
            operation()
        failures.append(str(caught.value))

    assert len(set(failures)) == 1
    for _ in range(3):
        with pytest.raises(ExecutionError, match="quarantined"):
            executor.run("must-not-reuse", workspace_bytes=1)


def test_destructor_sync_failure_quarantines_workspace(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    executor._inject_event_failure_for_test("synchronize")
    work = executor.run("discarded-failure", workspace_bytes=1024)

    del work
    gc.collect()

    with pytest.raises(ExecutionError, match="quarantined"):
        executor.run("must-not-reuse", workspace_bytes=1)
