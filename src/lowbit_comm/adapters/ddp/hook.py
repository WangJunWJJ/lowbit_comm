"""DDP communication hook over a precompiled lowbit_comm executable."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .state import CompressionSchema, GradientFeedbackState


_HOOK_COMPLETION_POOL = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="lowbit-ddp-hook",
)


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
    if not callable(reconstruct) and not callable(run_fused):
        raise TypeError("DDP Gradient EF requires executable.reconstruct_local()")
    world_size, compression_schema = _feedback_identity(
        executable,
        world_size=world_size,
        compression_schema=compression_schema,
    )

    def hook(_unused_state: Any, bucket: Any) -> Any:
        value = _bucket_buffer(bucket)
        transaction = state.prepare(
            _bucket_identity(bucket),
            value,
            world_size=world_size,
            compression_schema=compression_schema,
        )
        outer = future_factory()
        try:
            if callable(run_fused):
                work, local_restored = run_fused(transaction.prepared)
            else:
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
