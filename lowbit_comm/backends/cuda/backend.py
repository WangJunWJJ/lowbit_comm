"""Strict Phase 2 CUDA backend capability declarations and lowering."""

from __future__ import annotations

from lowbit_comm.api.intent import (
    CommunicationIntent,
    OutputSemantics,
    ShapeFamily,
    TensorSpec,
    _validate_communication_intent_graph,
)
from lowbit_comm.api.policy import (
    CollectiveKind,
    CompressionKind,
    StrategySpec,
    TopologyKind,
    _validate_strategy_graph,
)
from lowbit_comm.backends.cuda import loader
from lowbit_comm.backends.cuda.layout import (
    FullTensorLayout,
    build_fulltensor_layout,
)
from lowbit_comm.backends.cuda.plan import (
    CudaBackendPlan,
    _validate_phase2_request,
)
from lowbit_comm.backends.protocols import BackendCapability
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import _resolve_static_callable_member


_CUDA_DTYPES = frozenset({"fp16", "bf16"})
_CUDA_GROUP_SIZES = (16, 32, 64)
_CUDA_WORLD_SIZES = (2, 4)


class CudaBackend:
    """Lower only exact, safe FullTensor CUDA strategy contracts."""

    backend_id = "cuda"

    def capabilities(self) -> tuple[BackendCapability, ...]:
        """Return independent immutable snapshots of Phase 2 support."""
        strategies = (
            _native_strategy(),
            *(
                _int8_strategy(group_size)
                for group_size in _CUDA_GROUP_SIZES
            ),
        )
        return tuple(
            _capability(strategy, world_size)
            for strategy in strategies
            for world_size in _CUDA_WORLD_SIZES
        )

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> CudaBackendPlan:
        """Validate completely, then create one immutable native plan."""
        request = _validate_communication_intent_graph(intent)
        selected = _validate_strategy_graph(strategy)
        _validate_phase2_request(request, selected)
        layout = build_fulltensor_layout(
            numel=request.tensor.numel,
            dtype=request.tensor.dtype,
            world_size=request.world_size,
            group_size=selected.group_size or 16,
        )
        _validate_workspace_budget(selected, layout)
        request_snapshot = _snapshot_intent(request)
        strategy_snapshot = _snapshot_strategy(selected)
        module = loader.load_extension()
        create_plan = _resolve_static_callable_member(
            module,
            "create_fulltensor_plan",
            "CUDA extension must provide create_fulltensor_plan().",
        )
        try:
            native_plan = create_plan(
                _native_config(request_snapshot, strategy_snapshot, layout)
            )
        except CompileError:
            raise
        except Exception as error:
            raise CompileError("CUDA extension plan creation failed.") from error
        return CudaBackendPlan(
            request_snapshot,
            strategy_snapshot,
            layout,
            native_plan,
        )


def _capability(
    strategy: StrategySpec,
    world_size: int,
) -> BackendCapability:
    return BackendCapability(
        backend_id=CudaBackend.backend_id,
        strategy=strategy,
        output=OutputSemantics.FULL_TENSOR,
        min_world_size=world_size,
        max_world_size=world_size,
        supported_dtypes=frozenset(dtype for dtype in _CUDA_DTYPES),
        supports_async=True,
    )


def _native_strategy() -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.NONE,
        collective=CollectiveKind.NATIVE,
        topology=TopologyKind.BACKEND_DEFAULT,
    )


def _int8_strategy(group_size: int) -> StrategySpec:
    return StrategySpec(
        compression=CompressionKind.INT8,
        collective=CollectiveKind.COMPRESSED_ALL_GATHER_REDUCE,
        topology=TopologyKind.BACKEND_DEFAULT,
        group_size=group_size,
    )


def _validate_workspace_budget(
    strategy: StrategySpec,
    layout: FullTensorLayout,
) -> None:
    if strategy.workspace_budget_bytes is not None and (
        strategy.workspace_budget_bytes < layout.workspace_bytes
    ):
        raise CompileError("CUDA workspace budget is insufficient.")


def _snapshot_intent(intent: CommunicationIntent) -> CommunicationIntent:
    """Return an independent exact request snapshot for a compiled plan."""
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
    """Return an independent exact strategy snapshot for a compiled plan."""
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


def _native_config(
    intent: CommunicationIntent,
    strategy: StrategySpec,
    layout: FullTensorLayout,
) -> dict[str, object]:
    """Encode one exact native descriptor with no policy-time objects."""
    return {
        "accumulation_dtype": strategy.accumulation_dtype.value,
        "collective": strategy.collective.value,
        "compression": strategy.compression.value,
        "dtype": intent.tensor.dtype,
        "group_size": layout.group_size,
        "layout": layout,
        "numel": intent.tensor.numel,
        "rank": intent.rank,
        "reduction": intent.reduction.value,
        "workspace_bytes": layout.workspace_bytes,
        "world_size": intent.world_size,
    }


__all__ = ["CudaBackend"]
