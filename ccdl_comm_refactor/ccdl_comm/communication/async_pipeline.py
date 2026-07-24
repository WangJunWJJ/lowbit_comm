from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.gather_reduce import GatheredPayloads


class AsyncBucketPipeline:
    """Sequence async gather completion, reduce, feedback update, and Future completion."""

    def __init__(
        self,
        *,
        gather_work: Any,
        future: Any,
        dequantize_reduce: Callable[[GatheredPayloads], Any],
        update_feedback: Callable[[Any], None],
        advance_policy: Callable[[], None],
        completion_manager: CudaCompletionManager | Any | None = None,
    ) -> None:
        self._gather_work = gather_work
        self._future = future
        self._dequantize_reduce = dequantize_reduce
        self._update_feedback = update_feedback
        self._advance_policy = advance_policy
        self._completion_manager = completion_manager or CudaCompletionManager()

    def run(self) -> Any:
        inner_future = self._get_inner_future()
        if inner_future is not None and hasattr(inner_future, "then"):
            inner_future.then(self._complete)
        else:
            self._complete()
        return self._future

    def _get_inner_future(self) -> Any:
        get_future = getattr(self._gather_work, "get_future", None)
        if callable(get_future):
            return get_future()
        return None

    def _complete(self, _ignored: Any = None) -> Any:
        try:
            gathered = self._gather_work.wait()
            restored = self._dequantize_reduce(gathered)
            self._update_feedback(restored)
            self._advance_policy()
            completion = self._completion_manager.record_for(restored)
            completion.wait()
            self._future.set_result(restored)
            return restored
        except Exception as exc:
            set_exception = getattr(self._future, "set_exception", None)
            if callable(set_exception):
                set_exception(exc)
                return None
            raise
