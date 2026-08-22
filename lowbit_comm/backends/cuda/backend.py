"""Strict Phase 2 CUDA backend capability declarations and lowering."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

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
    ReducedShardLayout,
    build_fulltensor_layout,
    build_reduced_shard_layout,
)
from lowbit_comm.backends.cuda.plan import (
    CudaBackendPlan,
    CudaReducedShardPlan,
    _validate_phase2_request,
)
from lowbit_comm.api.result import ReducedShardMetadata
from lowbit_comm.backends.protocols import BackendCapability
from lowbit_comm.core.errors import CompileError
from lowbit_comm.core.plan import _resolve_static_callable_member


_CUDA_DTYPES = frozenset({"fp16", "bf16"})
_CUDA_GROUP_SIZES = (16, 32, 64)
_CUDA_WORLD_SIZES = (2, 4)


class _NativePlanAdapter(NamedTuple):
    execute_callable: Callable[..., object]

    def execute(self, value: object, *args: object) -> object:
        """Call one method bound at the trusted extension boundary."""
        return self.execute_callable(value, *args)


class CudaBackend:
    """Lower only exact, safe CUDA FullTensor and ReducedShard contracts."""

    backend_id = "cuda"

    def __init__(self, process_group: object | None = None) -> None:
        self._process_group = process_group

    def capabilities(self) -> tuple[BackendCapability, ...]:
        """Return independent immutable snapshots of Phase 2 support."""
        fulltensor_strategies = (
            _native_strategy(),
            *(
                _int8_strategy(group_size)
                for group_size in _CUDA_GROUP_SIZES
            ),
        )
        return (
            *(
                _capability(
                    strategy,
                    world_size,
                    OutputSemantics.FULL_TENSOR,
                )
                for strategy in fulltensor_strategies
                for world_size in _CUDA_WORLD_SIZES
            ),
            *(
                _capability(
                    _native_strategy(),
                    world_size,
                    OutputSemantics.REDUCED_SHARD,
                )
                for world_size in _CUDA_WORLD_SIZES
            ),
        )

    def lower(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> CudaBackendPlan | CudaReducedShardPlan:
        """Validate completely, then create one immutable native plan."""
        request = _validate_communication_intent_graph(intent)
        selected = _validate_strategy_graph(strategy)
        _validate_phase2_request(request, selected)
        layout = _build_layout(request, selected)
        _validate_workspace_budget(selected, layout)
        if self._process_group is None:
            raise CompileError(
                "CUDA backend requires an explicit ProcessGroup."
            )
        request_snapshot = _snapshot_intent(request)
        strategy_snapshot = _snapshot_strategy(selected)
        module = loader.load_extension()
        factory_name = _factory_name(request.output)
        create_plan = _resolve_static_callable_member(
            module,
            factory_name,
            f"CUDA extension must provide {factory_name}().",
        )
        try:
            native_plan = create_plan(
                _native_config(request_snapshot, strategy_snapshot, layout),
                self._process_group,
            )
            native_adapter = _NativePlanAdapter(
                _resolve_static_callable_member(
                    native_plan,
                    "execute",
                    "CUDA native plan must provide callable execute().",
                )
            )
        except CompileError:
            raise
        except Exception as error:
            raise CompileError(
                "CUDA extension plan creation failed."
            ) from error
        if request_snapshot.output is OutputSemantics.FULL_TENSOR:
            return CudaBackendPlan(
                request_snapshot,
                strategy_snapshot,
                layout,
                native_adapter,
            )
        return CudaReducedShardPlan(
            request_snapshot,
            strategy_snapshot,
            layout,
            _reduced_shard_metadata(request_snapshot, layout),
            native_adapter,
        )


def _capability(
    strategy: StrategySpec,
    world_size: int,
    output: OutputSemantics,
) -> BackendCapability:
    return BackendCapability(
        backend_id=CudaBackend.backend_id,
        strategy=strategy,
        output=output,
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
    layout: FullTensorLayout | ReducedShardLayout,
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
    layout: FullTensorLayout | ReducedShardLayout,
) -> dict[str, object]:
    """Encode one exact native descriptor with no policy-time objects."""
    if intent.output is OutputSemantics.REDUCED_SHARD:
        if type(layout) is not ReducedShardLayout:
            raise CompileError("CUDA ReducedShard layout is invalid.")
        config: dict[str, object] = {
            "accumulation_dtype": strategy.accumulation_dtype.value,
            "collective": strategy.collective.value,
            "compression": strategy.compression.value,
            "dtype": intent.tensor.dtype,
            "global_numel": layout.global_numel,
            "group_size": layout.group_size,
            "groups_per_shard": layout.groups_per_shard,
            "logical_shard_length": layout.logical_shard_length,
            "numel": intent.tensor.numel,
            "offset": layout.offset,
            "output_bytes": layout.output_bytes,
            "output_numel": layout.output_numel,
            "payload_bytes_per_destination": (
                layout.payload_bytes_per_destination
            ),
            "rank": intent.rank,
            "receive_payload_bytes": layout.receive_payload_bytes,
            "reduction": intent.reduction.value,
            "send_payload_bytes": layout.send_payload_bytes,
            "transport_shard_length": layout.transport_shard_length,
            "valid_length": layout.valid_length,
            "workspace_bytes": layout.workspace_bytes,
            "world_size": intent.world_size,
        }
        if strategy.error_feedback:
            config["gradient_error_feedback"] = True
        return config
    if type(layout) is not FullTensorLayout:
        raise CompileError("CUDA FullTensor layout is invalid.")
    config = {
        "accumulation_dtype": strategy.accumulation_dtype.value,
        "collective": strategy.collective.value,
        "compression": strategy.compression.value,
        "dtype": intent.tensor.dtype,
        "group_size": layout.group_size,
        "gathered_payload_bytes": layout.gathered_payload_bytes,
        "group_count": layout.group_count,
        "logical_numel": layout.logical_numel,
        "numel": intent.tensor.numel,
        "output_bytes": layout.output_bytes,
        "padded_numel": layout.padded_numel,
        "payload_bytes_per_rank": layout.payload_bytes_per_rank,
        "rank": intent.rank,
        "reduction": intent.reduction.value,
        "workspace_bytes": layout.workspace_bytes,
        "world_size": intent.world_size,
    }
    if strategy.error_feedback:
        config["gradient_error_feedback"] = True
    return config


def _build_layout(
    intent: CommunicationIntent,
    strategy: StrategySpec,
) -> FullTensorLayout | ReducedShardLayout:
    if intent.output is OutputSemantics.FULL_TENSOR:
        return build_fulltensor_layout(
            numel=intent.tensor.numel,
            dtype=intent.tensor.dtype,
            world_size=intent.world_size,
            compression=strategy.compression,
            group_size=strategy.group_size,
        )
    if intent.output is OutputSemantics.REDUCED_SHARD:
        return build_reduced_shard_layout(
            numel=intent.tensor.numel,
            dtype=intent.tensor.dtype,
            world_size=intent.world_size,
            compression=strategy.compression,
            group_size=strategy.group_size,
            rank=intent.rank,
        )
    raise CompileError("CUDA Phase 2 output semantics are unsupported.")


def _factory_name(output: OutputSemantics) -> str:
    if output is OutputSemantics.FULL_TENSOR:
        return "create_fulltensor_plan"
    if output is OutputSemantics.REDUCED_SHARD:
        return "create_reduced_shard_plan"
    raise CompileError("CUDA Phase 2 output semantics are unsupported.")


def _reduced_shard_metadata(
    intent: CommunicationIntent,
    layout: FullTensorLayout | ReducedShardLayout,
) -> ReducedShardMetadata:
    if type(layout) is not ReducedShardLayout:
        raise CompileError("CUDA ReducedShard layout is invalid.")
    return ReducedShardMetadata(
        global_shape=intent.tensor.shape,
        offset=layout.offset,
        valid_length=layout.valid_length,
        padded_length=layout.logical_shard_length,
        owner_rank=intent.rank,
    )


__all__ = ["CudaBackend"]
