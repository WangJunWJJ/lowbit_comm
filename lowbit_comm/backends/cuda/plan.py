"""Immutable CUDA lowering plans and Phase 2 request validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from lowbit_comm.api.intent import (
    CommunicationIntent,
    OutputSemantics,
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
    build_fulltensor_layout,
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
        if type(self.layout) is not FullTensorLayout:
            raise CompileError("CUDA backend plan layout is invalid.")
        expected_layout = build_fulltensor_layout(
            numel=request.tensor.numel,
            dtype=request.tensor.dtype,
            world_size=request.world_size,
            group_size=selected.group_size or 16,
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
    if intent.output is not OutputSemantics.FULL_TENSOR:
        raise CompileError("CUDA Phase 2 requires full-tensor output.")
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
    if strategy.collective is not CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE:
        raise CompileError(
            "CUDA INT8 strategy requires compressed all-gather reduce."
        )
    if strategy.group_size not in _PHASE2_GROUP_SIZES:
        raise CompileError("CUDA INT8 group size is unsupported.")


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


__all__ = ["CudaBackendPlan"]
