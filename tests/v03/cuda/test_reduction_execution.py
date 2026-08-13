from __future__ import annotations

from lowbit_comm.backends.cuda.executors import _finish_native_reduction


class _Handle:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _CudaEvent:
    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing is False

    def record(self, stream: object) -> None:
        del stream

    def query(self) -> bool:
        return True


class _Cuda:
    Event = _CudaEvent

    @staticmethod
    def current_stream(device: object) -> object:
        return device


class _Tensor:
    device = "cuda:0"

    def __init__(self, value: float) -> None:
        self.value = value
        self.divisors: list[int] = []

    def div_(self, divisor: int) -> "_Tensor":
        self.value /= divisor
        self.divisors.append(divisor)
        return self


def test_native_sum_waits_without_normalizing(monkeypatch) -> None:
    handle = _Handle()
    value = _Tensor(8.0)
    monkeypatch.setattr(
        "lowbit_comm.backends.cuda.executors.import_module",
        lambda name: type("Torch", (), {"cuda": _Cuda})(),
    )

    result = _finish_native_reduction(value, handle, divisor=1)

    assert result is value
    assert handle.waited
    assert value.value == 8.0
    assert value.divisors == []


def test_native_mean_normalizes_exactly_once(monkeypatch) -> None:
    handle = _Handle()
    value = _Tensor(8.0)
    monkeypatch.setattr(
        "lowbit_comm.backends.cuda.executors.import_module",
        lambda name: type("Torch", (), {"cuda": _Cuda})(),
    )

    result = _finish_native_reduction(value, handle, divisor=4)

    assert result is value
    assert handle.waited
    assert value.value == 2.0
    assert value.divisors == [4]
