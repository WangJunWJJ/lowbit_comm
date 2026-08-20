"""CUDA Work contract tests for terminal publication and launch identity."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib

import pytest

from lowbit_comm import ExecutionError


@pytest.fixture(scope="module")
def fake_extension():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native Work contract")
    return importlib.import_module("lowbit_comm._C")


def test_work_never_publishes_before_success(fake_extension) -> None:
    work = fake_extension.make_test_work(
        event_state="pending",
        value="result",
    )

    assert work.is_completed() is False
    with pytest.raises(ExecutionError, match="not completed"):
        work.result()


def test_successful_work_publishes_one_stable_result(fake_extension) -> None:
    value = object()
    work = fake_extension.make_test_work(
        event_state="success",
        value=value,
    )

    assert work.is_completed() is True
    assert work.wait() is value
    assert work.wait() is value
    assert work.result() is value


def test_failed_work_never_publishes_and_repeats_failure(
    fake_extension,
) -> None:
    work = fake_extension.make_test_work(
        event_state="failure",
        value="must-not-publish",
    )

    assert work.is_completed() is True
    for operation in (work.wait, work.wait, work.result):
        with pytest.raises(ExecutionError, match="test CUDA failure"):
            operation()


def test_concurrent_waiters_observe_one_terminal_result(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=1024,
    )
    work = executor.run("shared-result", workspace_bytes=1024)

    with ThreadPoolExecutor(max_workers=4) as threads:
        results = list(threads.map(lambda _: work.wait(), range(4)))

    assert results == ["shared-result"] * 4
    assert work.is_completed() is True


def test_launch_tokens_are_unique_and_monotonic_per_plan(
    fake_extension,
) -> None:
    first_executor = fake_extension.create_cuda_executor()
    second_executor = fake_extension.create_cuda_executor()

    first = first_executor.run("first").launch_token()
    second = first_executor.run("second").launch_token()
    other_plan = second_executor.run("other").launch_token()

    assert first.plan_id == second.plan_id
    assert second.sequence == first.sequence + 1
    assert other_plan.plan_id != first.plan_id
    assert (other_plan.plan_id, other_plan.sequence) != (
        first.plan_id,
        first.sequence,
    )


def test_monotonic_allocator_permanently_fails_closed_at_uint64_limit(
    fake_extension,
) -> None:
    uint64_max = (1 << 64) - 1
    allocator = fake_extension.make_test_monotonic_allocator(
        uint64_max - 2,
    )

    assert allocator.allocate() == uint64_max - 2
    assert allocator.allocate() == uint64_max - 1
    for _ in range(3):
        with pytest.raises(OverflowError, match="identity space is exhausted"):
            allocator.allocate()


def test_cuda_executor_sequence_exhaustion_precedes_every_side_effect(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor(
        workspace_capacity_bytes=64,
    )
    executor._exhaust_sequence_for_test()

    for workspace_bytes in (64, 0, 64):
        with pytest.raises(OverflowError, match="sequence space is exhausted"):
            executor.run("must-not-publish", workspace_bytes=workspace_bytes)

    assert executor._side_effect_counts_for_test() == {
        "allocation": 0,
        "workspace_acquire": 0,
        "transport_launch": 0,
        "kernel_launch": 0,
        "work_publish": 0,
    }


def test_runtime_boundary_rejects_legacy_python_completion(
    fake_extension,
) -> None:
    executor = fake_extension.create_cuda_executor()

    assert fake_extension.NATIVE_WORK_ABI_VERSION == 2
    assert not hasattr(fake_extension, "CompressedWork")
    with pytest.raises(TypeError):
        executor.run("result", completion=object())
    with pytest.raises(TypeError):
        executor.run("result", callback=lambda: "replacement")
