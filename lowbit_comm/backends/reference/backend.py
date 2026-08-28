"""Deterministic all-rank reference backend for contract oracles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from lowbit_comm.api.intent import (
    CommunicationIntent,
    CompletionMode,
    OutputSemantics,
    ReductionOp,
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
from lowbit_comm.api.result import (
    FullTensorResult,
    ReducedShardMetadata,
    ReducedShardResult,
)
from lowbit_comm.core.errors import (
    CompileError,
    ExecutionError,
)
from lowbit_comm.core.validation import _fresh_validate_exact
from lowbit_comm.runtime.work import CompletedWork, FailedWork


ReferenceValue = tuple[float, ...]
ReferenceResult = FullTensorResult[ReferenceValue] | ReducedShardResult[ReferenceValue]
ReferenceResults = tuple[ReferenceResult, ...]
RankValues = tuple[ReferenceValue, ...]


@dataclass(frozen=True, slots=True)
class ReferenceGroupPlan:
    """A compiled synchronous oracle that accepts all rank values."""

    intent: CommunicationIntent
    strategy: StrategySpec

    def __post_init__(self) -> None:
        _validate_compile_contract(self.intent, self.strategy)

    def execute_group(
        self,
        rank_values: object,
    ) -> CompletedWork[ReferenceResults] | FailedWork[ReferenceResults]:
        """Execute all-rank values and publish one terminal work object."""
        _validate_reference_group_plan(self)
        try:
            results = _execute_oracle(self.intent, rank_values)
        except ExecutionError as error:
            return FailedWork(error)
        return CompletedWork(results)


class ReferenceBackend:
    """Provide deterministic Python-float collective semantics."""

    backend_id = "reference"

    def compile_group(
        self,
        intent: CommunicationIntent,
        strategy: StrategySpec,
    ) -> ReferenceGroupPlan:
        """Compile one exact contract-only all-rank oracle plan."""
        _validate_compile_contract(intent, strategy)
        return ReferenceGroupPlan(
            _snapshot_intent(intent),
            _snapshot_strategy(strategy),
        )

    def execute_group(
        self,
        intent: CommunicationIntent,
        rank_values: RankValues,
    ) -> ReferenceResults:
        """Execute an all-rank contract oracle without selecting a strategy."""
        _validate_oracle_intent(intent)
        return _execute_oracle(_snapshot_intent(intent), rank_values)


def _validate_compile_contract(
    intent: object,
    strategy: object,
) -> None:
    """Reject non-exact values at the backend compile boundary."""
    _validate_oracle_intent(intent)
    _validate_oracle_strategy(strategy)


def _validate_oracle_intent(intent: object) -> CommunicationIntent:
    """Reject intents outside the synchronous reference contract."""
    if type(intent) is not CommunicationIntent:
        raise CompileError("Reference intent must be CommunicationIntent.")
    validated = _validate_communication_intent_graph(intent)
    if validated.completion is not CompletionMode.SYNC:
        raise CompileError("Reference group execution requires synchronous completion.")
    return validated


def _validate_oracle_strategy(strategy: object) -> StrategySpec:
    """Reject every strategy dimension the group oracle does not model."""
    validated = _validate_strategy_graph(strategy)
    if (
        validated.compression is not CompressionKind.NONE
        or validated.collective is not CollectiveKind.NATIVE
    ):
        raise CompileError(
            "Reference compression and collective must be NONE and NATIVE."
        )
    if validated.topology is not TopologyKind.BACKEND_DEFAULT:
        raise CompileError("Reference topology must be backend-default.")
    if validated.group_size is not None:
        raise CompileError("Reference group size must be unset.")
    if validated.accumulation_dtype is not AccumulationDType.FP32:
        raise CompileError("Reference accumulation must be FP32.")
    if validated.error_feedback:
        raise CompileError("Reference error feedback must be disabled.")
    if validated.parameter_error_feedback:
        raise CompileError("Reference parameter error feedback must be disabled.")
    if validated.overlap:
        raise CompileError("Reference overlap must be disabled.")
    if validated.workspace_budget_bytes is not None:
        raise CompileError("Reference workspace budget must be unset.")
    return validated


def _validate_reference_group_plan(plan: object) -> ReferenceGroupPlan:
    """Freshly validate an exact reference plan and its semantic graph."""
    return _fresh_validate_exact(
        plan,
        ReferenceGroupPlan,
        ReferenceGroupPlan.__post_init__,
        "Reference group plan graph is invalid.",
    )


def _snapshot_intent(intent: object) -> CommunicationIntent:
    """Build an independent exact intent graph after fresh validation."""
    source = _validate_oracle_intent(intent)
    return CommunicationIntent(
        tensor=TensorSpec(
            dtype=source.tensor.dtype,
            shape=tuple(dimension for dimension in source.tensor.shape),
        ),
        shape_family=ShapeFamily(
            max_numel=source.shape_family.max_numel,
            alignment=source.shape_family.alignment,
        ),
        reduction=source.reduction,
        output=source.output,
        completion=source.completion,
        world_size=source.world_size,
        rank=source.rank,
    )


def _snapshot_strategy(strategy: object) -> StrategySpec:
    """Build an independent exact strategy after fresh validation."""
    source = _validate_oracle_strategy(strategy)
    return StrategySpec(
        compression=source.compression,
        collective=source.collective,
        topology=source.topology,
        group_size=source.group_size,
        accumulation_dtype=source.accumulation_dtype,
        error_feedback=source.error_feedback,
        parameter_error_feedback=source.parameter_error_feedback,
        overlap=source.overlap,
        workspace_budget_bytes=source.workspace_budget_bytes,
    )


def _execute_oracle(
    intent: CommunicationIntent,
    rank_values: object,
) -> ReferenceResults:
    """Normalize unexpected oracle failures at the execution boundary."""
    try:
        return _execute_group(intent, rank_values)
    except ExecutionError:
        raise
    except Exception as error:
        raise ExecutionError("Reference group execution failed.") from error


def _execute_group(
    intent: CommunicationIntent,
    rank_values: object,
) -> ReferenceResults:
    """Validate, reduce, and materialize one all-rank oracle result."""
    values = _validate_rank_values(intent, rank_values)
    reduced = _reduce_values(intent, values)
    if intent.output is OutputSemantics.FULL_TENSOR:
        return tuple(FullTensorResult(value=reduced) for _ in range(intent.world_size))
    if intent.output is OutputSemantics.REDUCED_SHARD:
        return _reduced_shards(intent, reduced)
    raise ExecutionError("Reference output semantics are unsupported.")


def _validate_rank_values(
    intent: CommunicationIntent,
    rank_values: object,
) -> RankValues:
    """Return exact finite all-rank inputs or raise an execution error."""
    if type(rank_values) is not tuple:
        raise ExecutionError("Rank values must be an exact tuple.")
    if len(rank_values) != intent.world_size:
        raise ExecutionError("Reference rank count must equal intent world size.")
    for rank, value in enumerate(rank_values):
        if type(value) is not tuple:
            raise ExecutionError(f"Reference rank value {rank} must be an exact tuple.")
        if len(value) != intent.tensor.numel:
            raise ExecutionError(
                f"Reference rank {rank} tensor length must equal intent tensor numel."
            )
        for scalar in value:
            if type(scalar) is not float:
                raise ExecutionError(
                    "Reference input values must be exact Python float values."
                )
            if not isfinite(scalar):
                raise ExecutionError("Reference input values must be finite.")
    return rank_values


def _reduce_values(
    intent: CommunicationIntent,
    rank_values: RankValues,
) -> ReferenceValue:
    """Accumulate Python floats and apply mean exactly once when requested."""
    totals = [0.0] * intent.tensor.numel
    for rank_value in rank_values:
        for index in range(intent.tensor.numel):
            totals[index] += rank_value[index]
    if not all(isfinite(total) for total in totals):
        raise ExecutionError("Reference accumulation produced a non-finite reduction.")
    if intent.reduction is ReductionOp.SUM:
        return tuple(totals)
    if intent.reduction is ReductionOp.MEAN:
        divisor = float(intent.world_size)
        return tuple(total / divisor for total in totals)
    raise ExecutionError("Reference reduction operation is unsupported.")


def _reduced_shards(
    intent: CommunicationIntent,
    reduced: ReferenceValue,
) -> ReferenceResults:
    """Partition a reduced value into fixed-width contiguous rank shards."""
    numel = intent.tensor.numel
    padded_length = (
        0 if numel == 0 else (numel + intent.world_size - 1) // intent.world_size
    )
    results: list[ReferenceResult] = []
    for rank in range(intent.world_size):
        offset = min(rank * padded_length, numel)
        valid_length = min(padded_length, numel - offset)
        stop = offset + valid_length
        padding = (0.0,) * (padded_length - valid_length)
        metadata = ReducedShardMetadata(
            global_shape=intent.tensor.shape,
            offset=offset,
            valid_length=valid_length,
            padded_length=padded_length,
            owner_rank=rank,
        )
        results.append(
            ReducedShardResult(
                value=reduced[offset:stop] + padding,
                metadata=metadata,
            )
        )
    return tuple(results)
