"""DDP communication hook over a precompiled lowbit_comm executable."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .state import GradientFeedbackState


_HOOK_COMPLETION_POOL = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="lowbit-ddp-hook",
)


def create_ddp_hook(
    executable: Any,
    *,
    state: GradientFeedbackState,
    future_factory: type[Future[Any]] | Any = Future,
    bucket_type: type[Any] = Any,
    return_type: object = Any,
) -> Any:
    """Create a policy-free hook whose Future includes feedback completion."""

    reconstruct = getattr(executable, "reconstruct_local", None)
    if not callable(reconstruct):
        raise TypeError("DDP Gradient EF requires executable.reconstruct_local()")

    def hook(_unused_state: Any, bucket: Any) -> Any:
        value = _bucket_buffer(bucket)
        transaction = state.prepare(_bucket_identity(bucket), value)
        outer = future_factory()
        try:
            local_restored = reconstruct(transaction.prepared)
            work = executable.run(transaction.prepared)
        except BaseException as error:
            transaction.abort()
            outer.set_exception(error)
            return outer

        def finish() -> None:
            try:
                result = work.wait()
                transaction.commit(local_restored)
                outer.set_result(result)
            except BaseException as error:
                transaction.abort()
                outer.set_exception(error)

        _HOOK_COMPLETION_POOL.submit(finish)
        return outer

    hook.__annotations__ = {
        "_unused_state": Any,
        "bucket": bucket_type,
        "return": return_type,
    }
    return hook


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
