from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ccdl_comm.collectives.reduce_scatter import ReducedShard
from ccdl_comm.communication.cuda_completion import CudaCompletionManager


class AsyncShardPipeline:
    """Sequence async shard communication, local reduction, feedback, and Future completion."""

    def __init__(
        self,
        *,
        communication_work: Any,
        future: Any,
        reduce_shard: Callable[[Any], ReducedShard],
        update_feedback: Callable[[ReducedShard], None],
        advance_policy: Callable[[], None],
        completion_manager: CudaCompletionManager | Any | None = None,
        synchronize_completion: bool = True,
    ) -> None:
        self._communication_work = communication_work
        self._future = future
        self._reduce_shard = reduce_shard
        self._update_feedback = update_feedback
        self._advance_policy = advance_policy
        self._completion_manager = completion_manager or CudaCompletionManager()
        self._synchronize_completion = synchronize_completion

    def run(self) -> Any:
        inner_future = self._get_inner_future()
        if inner_future is not None and hasattr(inner_future, "then"):
            inner_future.then(self._complete)
        else:
            self._complete()
        return self._future

    def _get_inner_future(self) -> Any:
        get_future = getattr(self._communication_work, "get_future", None)
        if callable(get_future):
            return get_future()
        return None

    def _complete(self, _ignored: Any = None) -> ReducedShard | None:
        try:
            received = self._communication_work.wait()
            shard = self._mark_async(self._reduce_shard(received))
            self._update_feedback(shard)
            self._advance_policy()
            completion = self._completion_manager.record_for(shard.shard)
            completion.wait()
            if self._synchronize_completion:
                synchronize = getattr(completion, "synchronize", None)
                if callable(synchronize):
                    synchronize()
            self._future.set_result(shard)
            return shard
        except Exception as exc:
            set_exception = getattr(self._future, "set_exception", None)
            if callable(set_exception):
                set_exception(exc)
                return None
            raise

    def _mark_async(self, shard: ReducedShard) -> ReducedShard:
        metadata = dict(shard.metadata)
        metadata["async_completion"] = True
        return replace(shard, metadata=metadata)
