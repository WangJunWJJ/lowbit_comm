"""CUDA workspace lease lifecycle tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
from pathlib import Path
import threading

import pytest

from lowbit_comm import ExecutionError


ROOT = Path(__file__).resolve().parents[2]


def _concurrent_wait(work, barrier: threading.Barrier):
    barrier.wait(timeout=5.0)
    return work.wait()


def _concurrent_failed_wait(
    work,
    barrier: threading.Barrier,
) -> tuple[str, str]:
    barrier.wait(timeout=5.0)
    try:
        work.wait()
    except Exception as error:  # noqa: BLE001 - assert translated native type.
        return type(error).__name__, str(error)
    raise AssertionError("injected native event-sync failure did not raise")


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


def test_native_cuda_work_concurrent_waiters_finish_without_lost_wakeup(
    fake_extension,
) -> None:
    torch = importlib.import_module("torch")
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    torch.cuda._sleep(50_000_000)
    work = executor.run("shared", workspace_bytes=1024)
    barrier = threading.Barrier(9)

    with ThreadPoolExecutor(max_workers=8) as threads:
        futures = [
            threads.submit(_concurrent_wait, work, barrier)
            for _ in range(8)
        ]
        barrier.wait(timeout=5.0)
        results = [future.result(timeout=10.0) for future in futures]

    assert results == ["shared"] * 8
    assert work._synchronize_count_for_test() == 1
    assert executor.run("reused", workspace_bytes=1024).wait() == "reused"


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


def test_native_cuda_work_event_sync_failure_wakes_all_waiters(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    executor._inject_event_failure_for_test("synchronize")
    work = executor.run("must-fail", workspace_bytes=1024)
    barrier = threading.Barrier(9)

    with ThreadPoolExecutor(max_workers=8) as threads:
        futures = [
            threads.submit(_concurrent_failed_wait, work, barrier)
            for _ in range(8)
        ]
        barrier.wait(timeout=5.0)
        failures = [future.result(timeout=10.0) for future in futures]

    assert {failure[0] for failure in failures} == {"_CudaExecutionError"}
    assert len({failure[1] for failure in failures}) == 1
    assert "event synchronize failure" in failures[0][1]
    assert work._synchronize_count_for_test() == 1


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
