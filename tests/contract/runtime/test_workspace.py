from __future__ import annotations

import pytest

from lowbit_comm.runtime import (
    BudgetedWorkspacePool,
    CompletionWork,
    ManualCompletionEvent,
    WorkspaceBudgetExceeded,
    WorkspacePool,
)


def test_workspace_returns_to_pool_only_after_work_completion() -> None:
    pool: WorkspacePool[list[int]] = WorkspacePool()
    lease = pool.acquire(("bucket", 1024), lambda: [0] * 4)
    event = ManualCompletionEvent()
    work = CompletionWork(lease.value, event=event, resources=(lease,))

    assert pool.available(("bucket", 1024)) == 0
    event.complete()
    assert work.wait() == [0, 0, 0, 0]
    assert pool.available(("bucket", 1024)) == 1


def test_inflight_workspace_is_not_reused() -> None:
    pool: WorkspacePool[object] = WorkspacePool()
    first = pool.acquire("same", object)
    second = pool.acquire("same", object)

    assert first.value is not second.value
    first.release()
    with pytest.raises(RuntimeError, match="released"):
        _ = first.value
    second.release()


def test_release_is_idempotent() -> None:
    pool: WorkspacePool[object] = WorkspacePool()
    lease = pool.acquire("key", object)
    lease.release()
    lease.release()

    assert pool.available("key") == 1


def test_budgeted_pool_reuses_workspace_without_charging_twice() -> None:
    pool: BudgetedWorkspacePool[object] = BudgetedWorkspacePool(1024)
    allocations: list[object] = []

    first = pool.acquire(
        "send",
        512,
        lambda: allocations.append(object()) or allocations[-1],
    )
    value = first.value
    first.release()
    second = pool.acquire("send", 512, object)

    assert second.value is value
    assert pool.allocated_bytes == 512
    assert len(allocations) == 1


def test_budgeted_pool_rejects_new_allocation_above_budget() -> None:
    pool: BudgetedWorkspacePool[object] = BudgetedWorkspacePool(512)
    first = pool.acquire("send", 512, object)

    with pytest.raises(WorkspaceBudgetExceeded, match="budget"):
        pool.acquire("receive", 1, object)

    first.release()
