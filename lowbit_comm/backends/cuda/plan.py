"""Immutable CUDA lowering plans and Phase 2 request validation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from typing import TYPE_CHECKING, Callable, cast

from lowbit_comm.api.intent import (
    CommunicationIntent,
    OutputSemantics,
    ShapeFamily,
    TensorSpec,
    _validate_communication_intent_graph,
)
from lowbit_comm.api.policy import (
    AccumulationDType,
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
    _validate_strategy_graph,
)
from lowbit_comm.backends.cuda.layout import (
    FullTensorLayout,
    ReducedShardLayout,
    build_fulltensor_layout,
    build_reduced_shard_layout,
)
from lowbit_comm.api.result import (
    ReducedShardMetadata,
    ReducedShardResult,
)
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import _resolve_static_callable_member
from lowbit_comm.core.validation import _fresh_validate_exact

if TYPE_CHECKING:
    from lowbit_comm.runtime.work import CommunicationWork


_PHASE2_DTYPES = frozenset({"fp16", "bf16"})
_PHASE2_WORLD_SIZES = frozenset({2, 4})
_PHASE2_GROUP_SIZES = frozenset({16, 32, 64})


@dataclass(frozen=True, slots=True, eq=False)
class CudaBackendPlan:
    """One fully validated CUDA operation bound to a native plan object."""

    intent: CommunicationIntent
    strategy: StrategySpec
    layout: FullTensorLayout
    native_plan: object

    def __post_init__(self) -> None:
        request = _validate_communication_intent_graph(self.intent)
        selected = _validate_strategy_graph(self.strategy)
        _validate_phase2_request(request, selected)
        if request.output is not OutputSemantics.FULL_TENSOR:
            raise CompileError(
                "CUDA backend plan requires full-tensor output."
            )
        if type(self.layout) is not FullTensorLayout:
            raise CompileError("CUDA backend plan layout is invalid.")
        expected_layout = build_fulltensor_layout(
            numel=request.tensor.numel,
            dtype=request.tensor.dtype,
            world_size=request.world_size,
            compression=selected.compression,
            group_size=selected.group_size,
        )
        if self.layout != expected_layout:
            raise CompileError("CUDA backend plan layout is inconsistent.")
        if self.strategy.workspace_budget_bytes is not None and (
            self.strategy.workspace_budget_bytes < self.layout.workspace_bytes
        ):
            raise CompileError("CUDA workspace budget is insufficient.")
        _resolve_static_callable_member(
            self.native_plan,
            "execute",
            "CUDA native plan must provide callable execute().",
        )

    def execute(self, value: object) -> CommunicationWork[object]:
        """Delegate execution to the compiled native CUDA plan."""
        return _execute_cuda_plan(self, value)


def _validate_phase2_request(
    intent: CommunicationIntent,
    strategy: StrategySpec,
) -> None:
    """Reject every request outside the exact Phase 2 CUDA contract."""
    if intent.tensor.dtype not in _PHASE2_DTYPES:
        raise CompileError("CUDA Phase 2 dtype is unsupported.")
    if intent.world_size not in _PHASE2_WORLD_SIZES:
        raise CompileError("CUDA Phase 2 world size is unsupported.")
    if strategy.topology is not TopologyKind.BACKEND_DEFAULT:
        raise CompileError("CUDA Phase 2 topology is unsupported.")
    if strategy.accumulation_dtype is not AccumulationDType.FP32:
        raise CompileError("CUDA Phase 2 accumulation must be FP32.")
    if strategy.parameter_error_feedback:
        raise CompileError(
            "CUDA Phase 2 parameter error feedback is unsupported."
        )
    if strategy.error_feedback:
        raise CompileError("CUDA Phase 2 error feedback is unsupported.")
    if strategy.overlap:
        raise CompileError("CUDA Phase 2 overlap is unsupported.")
    if strategy.compression is CompressionKind.NONE:
        if strategy.collective is not CollectiveKind.NATIVE:
            raise CompileError(
                "CUDA native strategy must use NATIVE collective."
            )
        if strategy.group_size is not None:
            raise CompileError("CUDA native strategy cannot set a group size.")
        return
    if strategy.compression is not CompressionKind.INT8:
        raise CompileError("CUDA Phase 2 compression is unsupported.")
    if intent.output is OutputSemantics.FULL_TENSOR:
        if strategy.collective is not (
            CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE
        ):
            raise CompileError(
                "CUDA FullTensor INT8 strategy requires compressed "
                "all-gather reduce."
            )
    elif intent.output is OutputSemantics.REDUCED_SHARD:
        if strategy.collective is not CollectiveKind.COMPRESSED_REDUCE_SCATTER:
            raise CompileError(
                "CUDA ReducedShard INT8 strategy requires compressed "
                "reduce-scatter."
            )
    else:
        raise CompileError("CUDA Phase 2 output semantics are unsupported.")
    if strategy.group_size not in _PHASE2_GROUP_SIZES:
        raise CompileError("CUDA INT8 group size is unsupported.")


def _snapshot_intent(intent: CommunicationIntent) -> CommunicationIntent:
    """Return an independent exact request snapshot for one plan."""
    return CommunicationIntent(
        tensor=TensorSpec(
            dtype=intent.tensor.dtype,
            shape=tuple(dimension for dimension in intent.tensor.shape),
        ),
        shape_family=ShapeFamily(
            max_numel=intent.shape_family.max_numel,
            alignment=intent.shape_family.alignment,
        ),
        reduction=intent.reduction,
        output=intent.output,
        completion=intent.completion,
        world_size=intent.world_size,
        rank=intent.rank,
    )


def _snapshot_strategy(strategy: StrategySpec) -> StrategySpec:
    """Return an independent exact strategy snapshot for one plan."""
    return StrategySpec(
        compression=strategy.compression,
        collective=strategy.collective,
        topology=strategy.topology,
        group_size=strategy.group_size,
        accumulation_dtype=strategy.accumulation_dtype,
        error_feedback=strategy.error_feedback,
        parameter_error_feedback=strategy.parameter_error_feedback,
        overlap=strategy.overlap,
        workspace_budget_bytes=strategy.workspace_budget_bytes,
    )


def _validate_cuda_backend_plan(plan: object) -> CudaBackendPlan:
    """Freshly validate an exact CUDA plan before crossing its adapter."""
    return _fresh_validate_exact(
        plan,
        CudaBackendPlan,
        CudaBackendPlan.__post_init__,
        "CUDA backend plan graph is invalid.",
    )


def _execute_cuda_plan(
    plan: object,
    value: object,
) -> CommunicationWork[object]:
    """Perform no work locally; call the native compiled-plan adapter."""
    validated = _validate_cuda_backend_plan(plan)
    execute = _resolve_static_callable_member(
        validated.native_plan,
        "execute",
        "CUDA native plan must provide callable execute().",
    )
    return cast("CommunicationWork[object]", execute(value))


@dataclass(frozen=True, slots=True, eq=False)
class CudaReducedShardPlan:
    """One fully validated CUDA ReducedShard operation."""

    intent: CommunicationIntent
    strategy: StrategySpec
    layout: ReducedShardLayout
    metadata: ReducedShardMetadata
    native_plan: object

    def __post_init__(self) -> None:
        request = _validate_communication_intent_graph(self.intent)
        selected = _validate_strategy_graph(self.strategy)
        _validate_phase2_request(request, selected)
        if request.output is not OutputSemantics.REDUCED_SHARD:
            raise CompileError(
                "CUDA ReducedShard plan requires reduced-shard output."
            )
        if type(self.layout) is not ReducedShardLayout:
            raise CompileError("CUDA ReducedShard plan layout is invalid.")
        expected_layout = build_reduced_shard_layout(
            numel=request.tensor.numel,
            dtype=request.tensor.dtype,
            world_size=request.world_size,
            compression=selected.compression,
            group_size=selected.group_size,
            rank=request.rank,
        )
        if self.layout != expected_layout:
            raise CompileError(
                "CUDA ReducedShard plan layout is inconsistent."
            )
        _fresh_validate_exact(
            self.metadata,
            ReducedShardMetadata,
            ReducedShardMetadata.__post_init__,
            "CUDA ReducedShard metadata graph is invalid.",
        )
        expected_metadata = ReducedShardMetadata(
            global_shape=request.tensor.shape,
            offset=self.layout.offset,
            valid_length=self.layout.valid_length,
            padded_length=self.layout.logical_shard_length,
            owner_rank=request.rank,
        )
        if self.metadata != expected_metadata:
            raise CompileError("CUDA ReducedShard metadata is inconsistent.")
        if selected.workspace_budget_bytes is not None and (
            selected.workspace_budget_bytes < self.layout.workspace_bytes
        ):
            raise CompileError("CUDA workspace budget is insufficient.")
        _resolve_static_callable_member(
            self.native_plan,
            "execute",
            "CUDA native plan must provide callable execute().",
        )
        object.__setattr__(self, "intent", _snapshot_intent(request))
        object.__setattr__(self, "strategy", _snapshot_strategy(selected))
        object.__setattr__(self, "layout", expected_layout)
        object.__setattr__(self, "metadata", expected_metadata)

    def execute(
        self,
        value: object,
    ) -> CommunicationWork[ReducedShardResult[object]]:
        """Launch exactly one native work object for this owned shard."""
        return _execute_cuda_reduced_shard_plan(self, value)


class _ReducedShardWork:
    """Adapt one native work object to the immutable ReducedShard result."""

    def __init__(
        self,
        native_work: object,
        metadata: ReducedShardMetadata,
    ) -> None:
        self._is_completed: Callable[[], object] = (
            _resolve_static_callable_member(
                native_work,
                "is_completed",
                "CUDA native work must provide callable is_completed().",
            )
        )
        self._wait: Callable[[], object] = _resolve_static_callable_member(
            native_work,
            "wait",
            "CUDA native work must provide callable wait().",
        )
        self._metadata = metadata
        self._result: ReducedShardResult[object] | None = None
        self._failure: BaseException | None = None
        self._condition = Condition()
        self._wait_started = False
        self._terminal = False

    def is_completed(self) -> bool:
        """Delegate completion state directly to the native work object."""
        with self._condition:
            if self._terminal:
                return True
        return cast(bool, self._is_completed())

    def wait(self) -> ReducedShardResult[object]:
        """Cache exactly one terminal native result or execution failure."""
        with self._condition:
            if not self._wait_started:
                self._wait_started = True
                owns_wait = True
            else:
                owns_wait = False
                while not self._terminal:
                    self._condition.wait()
        if owns_wait:
            try:
                value = self._wait()
                result = ReducedShardResult(value, self._metadata)
                with self._condition:
                    self._result = result
                    self._terminal = True
                    self._condition.notify_all()
            except BaseException as failure:
                with self._condition:
                    self._failure = failure
                    self._terminal = True
                    self._condition.notify_all()
        if self._failure is not None:
            raise self._failure
        return cast(ReducedShardResult[object], self._result)

    def result(self) -> ReducedShardResult[object]:
        """Return the cached terminal result or the original failure."""
        return self.wait()


def _execute_cuda_reduced_shard_plan(
    plan: object,
    value: object,
) -> CommunicationWork[ReducedShardResult[object]]:
    """Launch one native work and bind its immutable ownership metadata."""
    validated = _validate_cuda_reduced_shard_plan(plan)
    execute = _resolve_static_callable_member(
        validated.native_plan,
        "execute",
        "CUDA native plan must provide callable execute().",
    )
    return _ReducedShardWork(execute(value), validated.metadata)


def _validate_cuda_reduced_shard_plan(
    plan: object,
) -> CudaReducedShardPlan:
    """Freshly validate an exact ReducedShard plan before execution."""
    return _fresh_validate_exact(
        plan,
        CudaReducedShardPlan,
        CudaReducedShardPlan.__post_init__,
        "CUDA ReducedShard plan graph is invalid.",
    )


__all__ = ["CudaBackendPlan", "CudaReducedShardPlan"]
