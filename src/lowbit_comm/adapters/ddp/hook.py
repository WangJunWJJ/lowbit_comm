"""DDP communication hook over a precompiled lowbit_comm executable."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from threading import RLock
from typing import Any

from lowbit_comm.runtime import (
    CompletionManager,
    CompletionPipeline,
    ImmediateCompletionEvent,
)

from .state import CompressionSchema, FeedbackTransaction, GradientFeedbackState


_HOOK_COMPLETION_MANAGER = CompletionManager()


class _CollectiveSequencer:
    def __init__(self) -> None:
        self.lock = RLock()
        self.pending: deque[tuple[Any, Any]] = deque()
        self.active = False


def create_ddp_hook(
    executable: Any,
    *,
    state: GradientFeedbackState,
    world_size: int | None = None,
    compression_schema: CompressionSchema | None = None,
    future_factory: type[Future[Any]] | Any = Future,
    bucket_type: type[Any] = Any,
    return_type: object = Any,
) -> Any:
    """Create a policy-free hook whose Future includes feedback completion."""

    reconstruct = getattr(executable, "reconstruct_local", None)
    run_fused = getattr(executable, "run_with_local_reconstruction", None)
    update_feedback = getattr(executable, "update_error_feedback", None)
    if not callable(reconstruct) and not callable(run_fused):
        raise TypeError("DDP Gradient EF requires executable.reconstruct_local()")
    world_size, compression_schema = _feedback_identity(
        executable,
        world_size=world_size,
        compression_schema=compression_schema,
    )
    serialize = bool(getattr(executable, "requires_collective_serialization", False))
    sequencer = (
        state.collective_sequencer(_CollectiveSequencer) if serialize else None
    )

    def submit(transaction: FeedbackTransaction) -> Any:
        try:
            if callable(run_fused):
                work, local_restored = run_fused(transaction.prepared)
            else:
                local_restored = reconstruct(transaction.prepared)
                work = executable.run(transaction.prepared)
        except BaseException as error:
            transaction.abort()
            outer = future_factory()
            outer.set_exception(error)
            return outer

        def finish(_pending: Any) -> Any:
            try:
                result = work.wait()
                transaction.commit(
                    getattr(local_restored, "value", local_restored),
                    updater=update_feedback if callable(update_feedback) else None,
                )
                return result
            except BaseException as error:
                transaction.abort()
                raise error

        pipeline = CompletionPipeline(None, resources=(work, local_restored))
        pipeline.add_stage("communication_work", _WorkCompletionEvent(work))
        pipeline.add_stage(
            "feedback_commit",
            ImmediateCompletionEvent(),
            action=finish,
        )
        return pipeline.get_future(
            future_factory,
            manager=_HOOK_COMPLETION_MANAGER,
        )

    def start_next() -> None:
        assert sequencer is not None
        with sequencer.lock:
            if not sequencer.pending:
                sequencer.active = False
                return
            start, outer = sequencer.pending[0]
        inner = start()

        def advance(completed: Any) -> Any:
            try:
                outer.set_result(_future_result(completed))
            except BaseException as error:
                outer.set_exception(error)
            finally:
                with sequencer.lock:
                    sequencer.pending.popleft()
                start_next()
            return completed

        _add_future_callback(inner, advance)

    def hook(_unused_state: Any, bucket: Any) -> Any:
        value = _bucket_buffer(bucket)
        transaction = state.prepare(
            _bucket_identity(bucket),
            value,
            world_size=world_size,
            compression_schema=compression_schema,
        )
        if not serialize:
            return submit(transaction)
        assert sequencer is not None
        outer = future_factory()
        with sequencer.lock:
            sequencer.pending.append((lambda: submit(transaction), outer))
            should_start = not sequencer.active
            if should_start:
                sequencer.active = True
        if should_start:
            start_next()
        return outer

    hook.__annotations__ = {
        "_unused_state": Any,
        "bucket": bucket_type,
        "return": return_type,
    }
    return hook


def _add_future_callback(future: Any, callback: Any) -> None:
    add_done_callback = getattr(future, "add_done_callback", None)
    if callable(add_done_callback):
        add_done_callback(callback)
        return
    then = getattr(future, "then", None)
    if callable(then):
        then(callback)
        return
    raise TypeError("future must provide add_done_callback() or then()")


def _future_result(future: Any) -> Any:
    result = getattr(future, "result", None)
    if callable(result):
        return result()
    value = getattr(future, "value", None)
    if callable(value):
        return value()
    wait = getattr(future, "wait", None)
    if callable(wait):
        return wait()
    raise TypeError("future must provide result(), value(), or wait()")


def _bucket_buffer(bucket: Any) -> Any:
    buffer = getattr(bucket, "buffer", None)
    if not callable(buffer):
        raise TypeError("DDP bucket must provide buffer()")
    return buffer()


def _bucket_identity(bucket: Any) -> Any:
    parameters = getattr(bucket, "parameters", None)
    if callable(parameters):
        identities = tuple(id(parameter) for parameter in parameters())
        if identities:
            return "parameters", identities
    index = getattr(bucket, "index", None)
    return index() if callable(index) else id(bucket)


class _WorkCompletionEvent:
    def __init__(self, work: Any) -> None:
        self._work = work

    def query(self) -> bool:
        query = getattr(self._work, "query", None)
        if callable(query):
            return bool(query())
        is_completed = getattr(self._work, "is_completed", None)
        return bool(is_completed()) if callable(is_completed) else True

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            raise NotImplementedError("communication work does not support timeout")
        self._work.wait()
        return True


def _feedback_identity(
    executable: Any,
    *,
    world_size: int | None,
    compression_schema: CompressionSchema | None,
) -> tuple[int, CompressionSchema]:
    lowered = getattr(executable, "lowered", None)
    context = getattr(lowered, "context", None)
    program = getattr(lowered, "program", None)
    wire = getattr(program, "wire", None)
    executor_kind = getattr(lowered, "executor_kind", None)
    resolved_world_size = world_size or getattr(context, "world_size", None)
    resolved_schema = compression_schema
    if resolved_schema is None and wire is not None and executor_kind is not None:
        resolved_schema = CompressionSchema(
            bit=wire.bit,
            group_size=wire.group_size,
            quant_type=wire.quant_type,
            compact=wire.compact,
            algorithm=executor_kind.value,
        )
    if not isinstance(resolved_world_size, int) or resolved_world_size <= 0:
        raise ValueError("DDP Gradient EF requires a positive compiled world_size")
    if not isinstance(resolved_schema, CompressionSchema):
        raise ValueError("DDP Gradient EF requires a compiled compression schema")
    return resolved_world_size, resolved_schema
