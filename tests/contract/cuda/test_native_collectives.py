from __future__ import annotations

from lowbit_comm.backends.cuda.native_collectives import CudaNativeCollectives


class Handle:
    def __init__(self) -> None:
        self.waited = False

    def is_completed(self) -> bool:
        return self.waited

    def wait(self) -> None:
        self.waited = True


class Dist:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str):
        def submit(*args: object, **kwargs: object) -> Handle:
            self.calls.append((name, args, kwargs))
            return Handle()

        return submit


def test_native_facade_submits_async_collective_with_bound_group() -> None:
    dist = Dist()
    facade = CudaNativeCollectives(process_group="group", dist=dist)
    tensor = object()

    work = facade.all_reduce(tensor, op="sum")

    assert dist.calls == [
        ("all_reduce", (tensor,), {"op": "sum", "group": "group", "async_op": True})
    ]
    assert work.query() is False
    assert work.wait() is tensor


def test_native_facade_covers_required_collective_primitives() -> None:
    required = {
        "all_reduce",
        "all_gather_into_tensor",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "broadcast",
        "reduce",
        "gather",
        "scatter",
        "barrier",
    }

    assert required <= set(dir(CudaNativeCollectives))
