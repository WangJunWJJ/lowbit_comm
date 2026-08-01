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
        synchronize_completion: bool = False,
        resources: tuple[Any, ...] = (),
        workspace_leases: tuple[Any, ...] = (),
    ) -> None:
        self._communication_work = communication_work
        self._future = future
        self._reduce_shard = reduce_shard
        self._update_feedback = update_feedback
        self._advance_policy = advance_policy
        self._completion_manager = completion_manager or CudaCompletionManager()
        self._synchronize_completion = synchronize_completion
        self._resources = tuple(resources)
        self._workspace_leases = list(workspace_leases)
        self._started = False

    def run(self) -> Any:
        inner_future = self._get_inner_future()
        if inner_future is not None and hasattr(inner_future, "then"):
            inner_future.then(self._complete)
        else:
            self._complete()
        self._started = True
        return self

    @property
    def resources(self) -> tuple[Any, ...]:
        """Return buffers retained until asynchronous completion."""

        return self._resources

    def wait(self) -> ReducedShard:
        """Wait for and return the reduced shard."""

        wait = getattr(self._future, "wait", None)
        if callable(wait):
            return wait()
        exception = getattr(self._future, "exception", None)
        if exception is not None:
            raise exception
        result = getattr(self._future, "result", None)
        if result is None:
            raise RuntimeError("asynchronous shard result is not complete")
        return result

    def query(self) -> bool:
        """Observe outer-future readiness without synchronizing."""

        done = getattr(self._future, "done", None)
        if callable(done):
            return bool(done())
        return getattr(self._future, "result", None) is not None or getattr(self._future, "exception", None) is not None

    def get_future(self) -> Any:
        """Return the underlying future for framework integration."""

        return self._future

    def then(self, callback: Callable[[Any], Any]) -> Any:
        """Delegate future chaining for framework compatibility."""

        then = getattr(self._future, "then", None)
        if not callable(then):
            raise AttributeError("underlying future does not support then()")
        return then(callback)

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
            self._release_workspace_leases(completion)
            if self._synchronize_completion:
                synchronize = getattr(completion, "synchronize", None)
                if callable(synchronize):
                    synchronize()
            self._future.set_result(shard)
            return shard
        except Exception as exc:
            if self._workspace_leases:
                buffer = getattr(self._workspace_leases[0], "buffer", None)
                completion = self._completion_manager.record_for(buffer)
                completion.wait()
                self._release_workspace_leases(completion)
            set_exception = getattr(self._future, "set_exception", None)
            if callable(set_exception):
                set_exception(exc)
                return None
            raise

    def _release_workspace_leases(self, completion: Any) -> None:
        while self._workspace_leases:
            lease = self._workspace_leases.pop(0)
            lease.release(completion=completion)

    def _mark_async(self, shard: ReducedShard) -> ReducedShard:
        metadata = dict(shard.metadata)
        metadata["async_completion"] = True
        return replace(shard, metadata=metadata)
