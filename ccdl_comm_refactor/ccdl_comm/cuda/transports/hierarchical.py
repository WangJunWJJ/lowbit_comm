"""Compile immutable hierarchical stages and execute them with event ordering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import reduce
from importlib import import_module
from operator import mul
from ccdl_comm.communication.cuda_completion import NoopCompletion
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.plan import CommunicationPlan, CompileContext
from ccdl_comm.stage import CommunicationStage
from ccdl_comm.work import CompletionWork


StageOperation = Callable[[object], object]
StageOperationFactory = Callable[[CommunicationStage, CompileContext], StageOperation]
GroupFactory = Callable[[tuple[int, ...]], object]
GroupMembers = Callable[[object], tuple[int, ...]]
StreamFactory = Callable[[CommunicationStage, CompileContext], object | None]
CompletionFactory = Callable[[object, object | None], object]


@dataclass(frozen=True, slots=True)
class StageExecution:
    """One launched stage result and its device-side completion dependency."""

    value: object
    completion: object | None = None
    resources: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", tuple(self.resources))


@dataclass(frozen=True, slots=True)
class CompiledStage:
    """A stage whose topology, layouts, stream, and operation are pre-bound."""

    name: str
    input_layout: str
    output_layout: str
    participants: tuple[int, ...]
    process_group: object
    operation: StageOperation
    stream: object | None = None
    completion_factory: CompletionFactory | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("compiled stage name must be non-empty")
        if not self.input_layout.strip() or not self.output_layout.strip():
            raise ValueError("compiled stage layouts must be non-empty")
        participants = tuple(self.participants)
        if not participants or len(set(participants)) != len(participants):
            raise ValueError("compiled stage participants must be non-empty and unique")
        if not callable(self.operation):
            raise TypeError("compiled stage operation must be callable")
        object.__setattr__(self, "participants", participants)

    def launch(self, value: object, dependency: object | None) -> StageExecution:
        """Launch after inserting a stream wait for the preceding stage event."""

        if dependency is not None:
            wait_stream = getattr(dependency, "wait_stream", None)
            if not callable(wait_stream):
                raise TypeError(
                    f"stage {self.name!r} dependency does not support wait_stream"
                )
            wait_stream(self.stream)
        result = self.operation(value)
        if isinstance(result, StageExecution):
            if result.completion is not None or self.completion_factory is None:
                return result
            return StageExecution(
                result.value,
                completion=self.completion_factory(result.value, self.stream),
                resources=result.resources,
            )
        completion = (
            NoopCompletion()
            if self.completion_factory is None
            else self.completion_factory(result, self.stream)
        )
        return StageExecution(result, completion=completion, resources=(result,))


class HierarchicalExecutor:
    """Execute a validated stage chain without host waits between stages."""

    def __init__(self, stages: tuple[CompiledStage, ...]) -> None:
        stages = tuple(stages)
        if not stages:
            raise ValueError("hierarchical executor requires at least one compiled stage")
        for previous, current in zip(stages, stages[1:], strict=False):
            if previous.output_layout != current.input_layout:
                raise ValueError(
                    f"stage layout mismatch: {previous.name!r} outputs "
                    f"{previous.output_layout!r}, but {current.name!r} consumes "
                    f"{current.input_layout!r}"
                )
        self.stages = stages

    @property
    def input_layout(self) -> str:
        return self.stages[0].input_layout

    @property
    def output_layout(self) -> str:
        return self.stages[-1].output_layout

    def run(self, tensor: object) -> CompletionWork[object]:
        """Launch the immutable chain and return work for the final stage event."""

        value = tensor
        dependency = None
        resources: list[object] = [tensor]
        for stage in self.stages:
            execution = stage.launch(value, dependency)
            value = execution.value
            dependency = execution.completion
            resources.extend(execution.resources)
        return CompletionWork(
            value,
            completion=dependency,
            resources=tuple(resources),
        )


def compile_hierarchical_stages(
    plan: CommunicationPlan,
    context: CompileContext,
    *,
    operation_factory: StageOperationFactory,
    group_members: GroupMembers | None = None,
    group_factory: GroupFactory | None = None,
    stream_factory: StreamFactory | None = None,
    completion_factory: CompletionFactory | None = None,
) -> HierarchicalExecutor:
    """Compile a three-stage local/inter/local hierarchy for the current rank."""

    if plan.strategy != "hierarchical":
        raise ValueError("hierarchical stage compilation requires strategy='hierarchical'")
    if not callable(operation_factory):
        raise TypeError("operation_factory must be callable")
    topology = _topology_for(context)
    resolve_members = group_members or _torch_group_members
    created_groups: dict[tuple[int, ...], object] = {}
    current_layout = "full"
    current_shape = tuple(context.shape)
    compiled: list[CompiledStage] = []

    for stage in plan.stages:
        expected_ranks = _expected_stage_ranks(
            stage,
            input_layout=current_layout,
            topology=topology,
        )
        process_group = _resolve_stage_group(
            stage,
            context,
            expected_ranks=expected_ranks,
            group_factory=group_factory,
            created_groups=created_groups,
        )
        actual_ranks = tuple(int(rank) for rank in resolve_members(process_group))
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"stage {stage.name!r} process group members {actual_ranks} do not "
                f"match expected ranks {expected_ranks}"
            )
        group_rank = expected_ranks.index(context.rank)
        stage_context = replace(
            context,
            rank=group_rank,
            world_size=len(expected_ranks),
            shape=current_shape,
            process_group=process_group,
            process_groups={},
        )
        operation = operation_factory(stage, stage_context)
        stream = None if stream_factory is None else stream_factory(stage, stage_context)
        compiled.append(
            CompiledStage(
                name=stage.name,
                input_layout=current_layout,
                output_layout=stage.output_layout,
                participants=expected_ranks,
                process_group=process_group,
                operation=operation,
                stream=stream,
                completion_factory=completion_factory,
            )
        )
        current_shape = _stage_output_shape(stage, current_shape, len(expected_ranks))
        current_layout = stage.output_layout

    if current_layout != plan.output_layout:
        raise ValueError(
            f"hierarchical final output layout {current_layout!r} does not match "
            f"plan output layout {plan.output_layout!r}"
        )
    return HierarchicalExecutor(tuple(compiled))


@dataclass(frozen=True, slots=True)
class _RankTopology:
    local_ranks: tuple[int, ...]
    inter_ranks: tuple[int, ...]


def _topology_for(context: CompileContext) -> _RankTopology:
    if context.local_rank is None or context.local_world_size is None:
        raise ValueError("hierarchical compilation requires local rank and world size")
    if context.node_id is None or context.node_count is None:
        raise ValueError("hierarchical compilation requires node id and node count")
    if context.world_size != context.local_world_size * context.node_count:
        raise ValueError("world size must equal local world size multiplied by node count")
    local_start = context.node_id * context.local_world_size
    local_ranks = tuple(range(local_start, local_start + context.local_world_size))
    if context.rank not in local_ranks or context.local_rank != context.rank - local_start:
        raise ValueError("global rank is inconsistent with node and local rank metadata")
    inter_ranks = tuple(
        node * context.local_world_size + context.local_rank
        for node in range(context.node_count)
    )
    return _RankTopology(local_ranks=local_ranks, inter_ranks=inter_ranks)


def _expected_stage_ranks(
    stage: CommunicationStage,
    *,
    input_layout: str,
    topology: _RankTopology,
) -> tuple[int, ...]:
    transition = (stage.collective, input_layout, stage.output_layout)
    if transition == ("reduce_scatter", "full", "shard"):
        return topology.local_ranks
    if transition == ("all_reduce", "shard", "shard"):
        return topology.inter_ranks
    if transition == ("all_gather", "shard", "full"):
        return topology.local_ranks
    raise UnsupportedCollective(
        f"hierarchical:{stage.name}",
        reason=(
            "unsupported stage transition "
            f"{stage.collective}:{input_layout}->{stage.output_layout}"
        ),
    )


def _resolve_stage_group(
    stage: CommunicationStage,
    context: CompileContext,
    *,
    expected_ranks: tuple[int, ...],
    group_factory: GroupFactory | None,
    created_groups: dict[tuple[int, ...], object],
) -> object:
    process_group = stage.process_group or context.process_groups.get(stage.name)
    if process_group is not None:
        return process_group
    if group_factory is None:
        raise ValueError(
            f"stage {stage.name!r} requires a compile-time process group or group factory"
        )
    if expected_ranks not in created_groups:
        created_groups[expected_ranks] = group_factory(expected_ranks)
    return created_groups[expected_ranks]


def _stage_output_shape(
    stage: CommunicationStage,
    input_shape: tuple[int, ...],
    group_size: int,
) -> tuple[int, ...]:
    if stage.collective == "reduce_scatter":
        numel = reduce(mul, input_shape, 1)
        return ((numel + group_size - 1) // group_size,)
    return input_shape


def _torch_group_members(group: object) -> tuple[int, ...]:
    dist = import_module("torch.distributed")
    getter = getattr(dist, "get_process_group_ranks", None)
    if not callable(getter):
        raise RuntimeError(
            "torch.distributed.get_process_group_ranks is required for hierarchical compilation"
        )
    return tuple(int(rank) for rank in getter(group))
