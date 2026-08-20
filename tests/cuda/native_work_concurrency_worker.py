"""Isolated deterministic native CudaWork waiter interleaving."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("success", "failure"),
        required=True,
    )
    return parser.parse_args()


def _wait_for_state(work, key: str, expected) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = work._wait_latch_state_for_test()
        if state[key] == expected:
            return
        if key == "losers_arrived" and state[key] >= expected:
            return
        time.sleep(0.001)
    raise AssertionError(f"wait latch did not reach {key}={expected}")


def _wait_success(work, barrier: threading.Barrier):
    barrier.wait(timeout=5.0)
    return work.wait()


def _wait_failure(work, barrier: threading.Barrier) -> tuple[str, str]:
    barrier.wait(timeout=5.0)
    try:
        work.wait()
    except Exception as error:  # noqa: BLE001 - validate pybind translation.
        return type(error).__name__, str(error)
    raise AssertionError("native event-sync failure did not raise")


def main() -> None:
    args = _parse_args()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    extension = importlib.import_module("lowbit_comm._C")
    executor = extension.create_cuda_executor(workspace_capacity_bytes=1024)
    if args.mode == "failure":
        executor._inject_event_failure_for_test("synchronize")
    work = executor.run("shared", workspace_bytes=1024)
    work._enable_wait_latch_for_test(7)
    barrier = threading.Barrier(8)
    operation = _wait_success if args.mode == "success" else _wait_failure

    with ThreadPoolExecutor(max_workers=8) as threads:
        futures = [threads.submit(operation, work, barrier) for _ in range(8)]
        _wait_for_state(work, "owner_completion_blocked", True)
        print("LATCH owner-blocked", flush=True)
        _wait_for_state(work, "losers_arrived", 7)
        print("LATCH losers-arrived", flush=True)
        work._allow_completion_for_test()
        _wait_for_state(work, "terminal_publish_attempted", True)
        print("LATCH terminal-attempted", flush=True)
        work._release_losers_for_test()
        print("LATCH losers-released", flush=True)
        results = [future.result(timeout=5.0) for future in futures]
        print("LATCH futures-complete", flush=True)

    assert work._synchronize_count_for_test() == 1
    if args.mode == "success":
        assert results == ["shared"] * 8
    else:
        assert {result[0] for result in results} == {"_CudaExecutionError"}
        assert len({result[1] for result in results}) == 1
        assert "event synchronize failure" in results[0][1]
    print(f"NATIVE_WORK_CONCURRENCY_OK mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
