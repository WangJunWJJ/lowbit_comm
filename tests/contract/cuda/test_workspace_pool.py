from __future__ import annotations

from lowbit_comm.backends.cuda.workspace import CudaWorkspaceManager, CudaWorkspaceKey
from lowbit_comm.core import BufferSpec, WorkspaceRole
from lowbit_comm.runtime import BudgetedWorkspacePool
from lowbit_comm.runtime import CompletionOutcome, CompletionWork, ManualCompletionEvent


def test_cuda_workspace_key_separates_quantization_and_bucket_layouts() -> None:
    first = CudaWorkspaceKey(
        role=WorkspaceRole.SEND,
        shape=(4, 80),
        dtype="uint8",
        device="cuda:0",
        world_size=4,
        bit=8,
        group_size=64,
        compact=False,
    )
    second = CudaWorkspaceKey(
        role=WorkspaceRole.SEND,
        shape=(4, 80),
        dtype="uint8",
        device="cuda:0",
        world_size=4,
        bit=4,
        group_size=64,
        compact=False,
    )

    assert first != second


def test_cuda_workspace_manager_reuses_released_internal_buffer() -> None:
    pool: BudgetedWorkspacePool[object] = BudgetedWorkspacePool(1024)
    manager = CudaWorkspaceManager(
        pool=pool,
        world_size=2,
        bit=8,
        group_size=64,
        compact=False,
    )
    spec = BufferSpec(WorkspaceRole.SEND, (128,), "uint8", 128)
    allocations: list[object] = []

    first = manager.acquire(
        spec,
        device="cuda:0",
        allocator=lambda: allocations.append(object()) or allocations[-1],
    )
    value = first.value
    first.release()
    second = manager.acquire(spec, device="cuda:0", allocator=object)

    assert second.value is value
    assert len(allocations) == 1


def test_cuda_workspace_manager_uses_bound_external_pool() -> None:
    external: BudgetedWorkspacePool[object] = BudgetedWorkspacePool()
    manager = CudaWorkspaceManager(
        pool=external,
        world_size=2,
        bit=8,
        group_size=64,
        compact=True,
    )
    spec = BufferSpec(WorkspaceRole.RECEIVE, (64,), "uint8", 64)

    lease = manager.acquire(spec, device="cuda:1", allocator=object)
    lease.release()

    assert external.available(manager.key(spec, "cuda:1")) == 1


def test_cuda_workspace_is_not_reused_until_output_event_completes() -> None:
    pool: BudgetedWorkspacePool[object] = BudgetedWorkspacePool()
    manager = CudaWorkspaceManager(
        pool=pool,
        world_size=2,
        bit=8,
        group_size=64,
        compact=False,
    )
    spec = BufferSpec(WorkspaceRole.SEND, (64,), "uint8", 64)
    first = manager.acquire(spec, device="cuda:0", allocator=object)
    first_value = first.value
    output_ready = ManualCompletionEvent()
    work = CompletionWork(
        None,
        complete=lambda: CompletionOutcome(None, output_ready),
        resources=(first,),
    )
    second = manager.acquire(spec, device="cuda:0", allocator=object)

    assert second.value is not first_value
    assert work.query() is False
    output_ready.complete()
    assert work.query() is True
    third = manager.acquire(spec, device="cuda:0", allocator=object)
    assert third.value is first_value

    second.release()
    third.release()
