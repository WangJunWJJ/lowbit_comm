"""Compiled native Torch/NCCL collective executors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any

from ccdl_comm.execution_info import ExecutionCounters, ExecutionInfo
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.work import CompletionWork, bind_execution_work


NativeOperation = Callable[["NativeCollectiveInput"], tuple[object, object | None]]
NativeBuilder = Callable[[CommunicationPlan, CompileContext, Any], NativeOperation]


@dataclass(frozen=True, slots=True)
class NativeCollectiveInput:
    """Runtime tensors supplied to one precompiled native collective.

    Attributes:
        tensor: Primary input or caller-owned output tensor.
        input_tensors: Rank-ordered input tensors for list collectives.
        output_tensors: Rank-ordered caller-owned output tensors.
    """

    tensor: object | None = None
    input_tensors: tuple[object, ...] = ()
    output_tensors: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_tensors", tuple(self.input_tensors))
        object.__setattr__(self, "output_tensors", tuple(self.output_tensors))


class CudaNativeCollectiveExecutor:
    """Execute a process-group-bound native collective without replanning."""

    def __init__(
        self,
        operation: NativeOperation,
        execution_info: ExecutionInfo,
    ) -> None:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(execution_info, ExecutionInfo):
            raise TypeError("execution_info must be an ExecutionInfo")
        self._operation = operation
        self.execution_info = execution_info
        self.execution_counters = ExecutionCounters()
        self.last_handle: object | None = None

    def run(self, value: object):
        """Run the bound collective and retain all asynchronous resources."""

        self.execution_counters._record_run()
        invocation = (
            value
            if isinstance(value, NativeCollectiveInput)
            else NativeCollectiveInput(tensor=value)
        )
        try:
            result, handle = self._operation(invocation)
            self.last_handle = handle
            work = CompletionWork(
                result,
                handle=handle,
                resources=_resources(invocation, result),
            )
            return bind_execution_work(
                work,
                self.execution_info,
                self.execution_counters,
            )
        except BaseException:
            self.execution_counters._record_failed()
            raise


def compile_native_collective(
    plan: CommunicationPlan,
    context: CompileContext,
    *,
    dist: Any | None = None,
) -> CudaNativeCollectiveExecutor:
    """Compile one native collective with immutable group and control options."""

    if not isinstance(plan, CommunicationPlan):
        raise TypeError("plan must be a CommunicationPlan")
    if not isinstance(context, CompileContext):
        raise TypeError("context must be a CompileContext")
    if plan.strategy != "native_nccl":
        raise ValueError("native collective compiler requires strategy='native_nccl'")
    try:
        builder = NATIVE_BUILDERS[plan.collective]
    except KeyError as exc:
        raise ValueError(
            f"unsupported native collective: {plan.collective!r}"
        ) from exc
    if plan.root >= context.world_size:
        raise ValueError(
            f"root must be smaller than world_size ({plan.root} >= {context.world_size})"
        )
    active_dist = dist if dist is not None else import_module("torch.distributed")
    operation = builder(plan, context, active_dist)
    return CudaNativeCollectiveExecutor(
        operation,
        _execution_info(plan, context),
    )


def build_native_all_reduce(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    reduce_op = _resolve_reduce_op(plan.reduce_op, dist)

    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        tensor = _require_tensor(value, "all_reduce")
        handle = dist.all_reduce(
            tensor,
            op=reduce_op,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return tensor, handle

    return operation


def build_native_all_gather(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        tensor = _require_tensor(value, "all_gather")
        outputs = list(value.output_tensors)
        if not outputs:
            outputs = [
                tensor.new_empty(tuple(getattr(tensor, "shape", ())))
                for _ in range(context.world_size)
            ]
        _require_world_sized(outputs, context, "all_gather output_tensors")
        handle = dist.all_gather(
            outputs,
            tensor,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return tuple(outputs), handle

    return operation


def build_native_reduce_scatter(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    reduce_op = _resolve_reduce_op(plan.reduce_op, dist)

    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        output = _require_tensor(value, "reduce_scatter")
        inputs = list(value.input_tensors)
        _require_world_sized(inputs, context, "reduce_scatter input_tensors")
        handle = dist.reduce_scatter(
            output,
            inputs,
            op=reduce_op,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return output, handle

    return operation


def build_native_all_to_all(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        inputs = list(value.input_tensors)
        outputs = list(value.output_tensors)
        _require_world_sized(inputs, context, "all_to_all input_tensors")
        _require_world_sized(outputs, context, "all_to_all output_tensors")
        handle = dist.all_to_all(
            outputs,
            inputs,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return tuple(outputs), handle

    return operation


def build_native_broadcast(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        tensor = _require_tensor(value, "broadcast")
        handle = dist.broadcast(
            tensor,
            src=plan.root,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return tensor, handle

    return operation


def build_native_reduce(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    reduce_op = _resolve_reduce_op(plan.reduce_op, dist)

    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        tensor = _require_tensor(value, "reduce")
        handle = dist.reduce(
            tensor,
            dst=plan.root,
            op=reduce_op,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return tensor, handle

    return operation


def build_native_gather(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        tensor = _require_tensor(value, "gather")
        outputs = list(value.output_tensors)
        if context.rank == plan.root:
            if not outputs:
                outputs = [
                    tensor.new_empty(tuple(getattr(tensor, "shape", ())))
                    for _ in range(context.world_size)
                ]
            _require_world_sized(outputs, context, "gather output_tensors")
            gather_list = outputs
            result: object = tuple(outputs)
        else:
            if outputs:
                raise ValueError(
                    "gather output_tensors are valid only on the root rank"
                )
            gather_list = None
            result = None
        handle = dist.gather(
            tensor,
            gather_list=gather_list,
            dst=plan.root,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return result, handle

    return operation


def build_native_scatter(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        output = _require_tensor(value, "scatter")
        inputs = list(value.input_tensors)
        if context.rank == plan.root:
            _require_world_sized(inputs, context, "scatter input_tensors")
            scatter_list = inputs
        else:
            if inputs:
                raise ValueError(
                    "scatter input_tensors are valid only on the root rank"
                )
            scatter_list = None
        handle = dist.scatter(
            output,
            scatter_list=scatter_list,
            src=plan.root,
            group=context.process_group,
            async_op=plan.async_op,
        )
        return output, handle

    return operation


def build_native_barrier(
    plan: CommunicationPlan,
    context: CompileContext,
    dist: Any,
) -> NativeOperation:
    device_ids = _device_ids(context.device)

    def operation(value: NativeCollectiveInput) -> tuple[object, object | None]:
        del value
        handle = dist.barrier(
            group=context.process_group,
            async_op=plan.async_op,
            device_ids=device_ids,
        )
        return None, handle

    return operation


NATIVE_BUILDERS: dict[str, NativeBuilder] = {
    "all_reduce": build_native_all_reduce,
    "all_gather": build_native_all_gather,
    "reduce_scatter": build_native_reduce_scatter,
    "all_to_all": build_native_all_to_all,
    "broadcast": build_native_broadcast,
    "reduce": build_native_reduce,
    "gather": build_native_gather,
    "scatter": build_native_scatter,
    "barrier": build_native_barrier,
}


def _resolve_reduce_op(name: str, dist: Any) -> object:
    normalized = name.strip().lower()
    attribute = {
        "sum": "SUM",
        "mean": "AVG",
        "avg": "AVG",
        "product": "PRODUCT",
        "min": "MIN",
        "max": "MAX",
    }.get(normalized)
    if attribute is None:
        raise ValueError(f"unsupported reduce_op: {name!r}")
    try:
        return getattr(dist.ReduceOp, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"torch.distributed does not support reduce_op {name!r}"
        ) from exc


def _require_tensor(value: NativeCollectiveInput, collective: str) -> object:
    if value.tensor is None:
        raise ValueError(f"{collective} requires a primary tensor")
    return value.tensor


def _require_world_sized(
    tensors: list[object],
    context: CompileContext,
    name: str,
) -> None:
    if len(tensors) != context.world_size:
        raise ValueError(
            f"{name} must contain world_size tensors "
            f"({len(tensors)} != {context.world_size})"
        )


def _device_ids(device: str) -> list[int] | None:
    _, separator, suffix = device.partition(":")
    if not separator:
        return None
    try:
        return [int(suffix)]
    except ValueError:
        return None


def _resources(
    value: NativeCollectiveInput,
    result: object,
) -> tuple[object, ...]:
    resources = [
        item
        for item in (
            value.tensor,
            *value.input_tensors,
            *value.output_tensors,
            result,
        )
        if item is not None
    ]
    return tuple(resources)


def _execution_info(
    plan: CommunicationPlan,
    context: CompileContext,
) -> ExecutionInfo:
    numel = reduce(mul, context.shape, 1)
    original_bytes = numel * _dtype_bytes(context.dtype)
    fast_path = (
        "cuda_native_nccl"
        if plan.collective == "all_reduce"
        else f"cuda_native_nccl_{plan.collective}"
    )
    return ExecutionInfo(
        requested_strategy=plan.strategy,
        executed_strategy=plan.strategy,
        backend="cuda",
        fallback_used=False,
        fallback_reason=None,
        stage_names=(),
        original_bytes=original_bytes,
        compressed_bytes=original_bytes,
        compression_ratio=1.0,
        workspace_cache_hit=False,
        async_capable=True,
        fast_path=fast_path,
        details={
            "collective": plan.collective,
            "device": context.device,
            "dtype": context.dtype,
            "world_size": context.world_size,
            "root": plan.root,
            "reduce_op": plan.reduce_op,
        },
    )


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.strip().lower().removeprefix("torch.")
    if normalized in {"fp16", "float16", "half", "bf16", "bfloat16"}:
        return 2
    if normalized in {"fp64", "float64", "double"}:
        return 8
    return 4


__all__ = [
    "CudaNativeCollectiveExecutor",
    "NATIVE_BUILDERS",
    "NativeCollectiveInput",
    "build_native_all_gather",
    "build_native_all_reduce",
    "build_native_all_to_all",
    "build_native_barrier",
    "build_native_broadcast",
    "build_native_gather",
    "build_native_reduce",
    "build_native_reduce_scatter",
    "build_native_scatter",
    "compile_native_collective",
]
