"""Pre-bound zero-policy facade over native torch.distributed collectives."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from lowbit_comm.runtime import CompletionWork


class _HandleEvent:
    def __init__(self, handle: object) -> None:
        self._handle = handle

    def query(self) -> bool:
        query = getattr(self._handle, "is_completed", None)
        return bool(query()) if callable(query) else False

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        self._handle.wait()  # type: ignore[attr-defined]
        return True


class CudaNativeCollectives:
    """Submit explicit native collectives without registry or fallback logic."""

    def __init__(self, *, process_group: object | None = None, dist: Any | None = None):
        self._group = process_group
        self._dist = dist or import_module("torch.distributed")

    def all_reduce(self, tensor: Any, *, op: object = None) -> CompletionWork[Any]:
        return self._submit("all_reduce", tensor, op=op, result=tensor)

    def all_gather_into_tensor(self, output: Any, tensor: Any) -> CompletionWork[Any]:
        return self._submit("all_gather_into_tensor", output, tensor, result=output)

    def reduce_scatter_tensor(self, output: Any, tensor: Any, *, op: object = None) -> CompletionWork[Any]:
        return self._submit("reduce_scatter_tensor", output, tensor, op=op, result=output)

    def all_to_all_single(self, output: Any, tensor: Any) -> CompletionWork[Any]:
        return self._submit("all_to_all_single", output, tensor, result=output)

    def broadcast(self, tensor: Any, *, src: int) -> CompletionWork[Any]:
        return self._submit("broadcast", tensor, src=src, result=tensor)

    def reduce(self, tensor: Any, *, dst: int, op: object = None) -> CompletionWork[Any]:
        return self._submit("reduce", tensor, dst=dst, op=op, result=tensor)

    def gather(
        self, tensor: Any, *, gather_list: list[Any] | None, dst: int
    ) -> CompletionWork[list[Any] | None]:
        return self._submit(
            "gather",
            tensor,
            gather_list=gather_list,
            dst=dst,
            result=gather_list,
        )

    def scatter(
        self, tensor: Any, *, scatter_list: list[Any] | None, src: int
    ) -> CompletionWork[Any]:
        return self._submit(
            "scatter",
            tensor,
            scatter_list=scatter_list,
            src=src,
            result=tensor,
        )

    def barrier(self) -> CompletionWork[None]:
        return self._submit("barrier", result=None)

    def _submit(self, operation: str, *args: Any, result: Any, **kwargs: Any) -> CompletionWork[Any]:
        kwargs = {name: value for name, value in kwargs.items() if value is not None}
        handle = getattr(self._dist, operation)(
            *args,
            **kwargs,
            group=self._group,
            async_op=True,
        )
        return CompletionWork(
            result,
            event=_HandleEvent(handle),
            resources=args + tuple(kwargs.values()),
        )
