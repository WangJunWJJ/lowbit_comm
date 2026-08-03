from __future__ import annotations

from dataclasses import replace

import pytest

from ccdl_comm import CommunicationPlan, CommunicationStage, CompileContext, CompressionConfig
from ccdl_comm.cuda.transports.hierarchical import (
    CompiledStage,
    HierarchicalExecutor,
    StageExecution,
    compile_hierarchical_stages,
)


CONFIG = CompressionConfig(bit=8, group_size=64)


class FakeGroup:
    def __init__(self, ranks: tuple[int, ...]) -> None:
        self.ranks = ranks


class FakeCompletion:
    def __init__(self, name: str, calls: list[tuple[object, ...]]) -> None:
        self.name = name
        self.calls = calls

    def wait_stream(self, stream: object) -> None:
        self.calls.append(("wait_stream", self.name, stream))

    def wait(self) -> None:
        self.calls.append(("wait", self.name))

    def query(self) -> bool:
        return True


def _plan(groups: dict[str, FakeGroup]) -> CommunicationPlan:
    return CommunicationPlan(
        "all_reduce",
        "hierarchical",
        compression=CONFIG,
        stages=(
            CommunicationStage(
                "intra_reduce_scatter",
                "reduce_scatter",
                "compressed",
                compression=CONFIG,
                process_group=groups["intra_reduce_scatter"],
                output_layout="shard",
                async_op=False,
            ),
            CommunicationStage(
                "inter_ring",
                "all_reduce",
                "topology",
                compression=CONFIG,
                process_group=groups["inter_ring"],
                output_layout="shard",
                async_op=False,
            ),
            CommunicationStage(
                "restore_full",
                "all_gather",
                "native_nccl",
                process_group=groups["restore_full"],
                output_layout="full",
                async_op=False,
            ),
        ),
        async_op=False,
    )


def _context(rank: int) -> CompileContext:
    return CompileContext(
        rank=rank,
        world_size=8,
        device=f"cuda:{rank % 4}",
        shape=(4096,),
        dtype="fp16",
        local_rank=rank % 4,
        local_world_size=4,
        node_id=rank // 4,
        node_count=2,
    )


@pytest.mark.parametrize("rank", range(8))
def test_fake_eight_rank_stage_groups_and_layouts_are_compiled_per_rank(rank: int) -> None:
    local_start = (rank // 4) * 4
    local_ranks = tuple(range(local_start, local_start + 4))
    inter_ranks = (rank % 4, rank % 4 + 4)
    groups = {
        "intra_reduce_scatter": FakeGroup(local_ranks),
        "inter_ring": FakeGroup(inter_ranks),
        "restore_full": FakeGroup(local_ranks),
    }
    compiled_contexts = []

    def operation_factory(stage, stage_context):
        compiled_contexts.append((stage.name, stage_context))
        return lambda value: value

    executor = compile_hierarchical_stages(
        _plan(groups),
        _context(rank),
        operation_factory=operation_factory,
        group_members=lambda group: group.ranks,
    )

    assert tuple(stage.name for stage in executor.stages) == (
        "intra_reduce_scatter",
        "inter_ring",
        "restore_full",
    )
    assert tuple(stage.input_layout for stage in executor.stages) == (
        "full",
        "shard",
        "shard",
    )
    assert tuple(stage.output_layout for stage in executor.stages) == (
        "shard",
        "shard",
        "full",
    )
    assert tuple(stage.participants for stage in executor.stages) == (
        local_ranks,
        inter_ranks,
        local_ranks,
    )
    assert [item[0] for item in compiled_contexts] == [
        "intra_reduce_scatter",
        "inter_ring",
        "restore_full",
    ]
    assert [item[1].rank for item in compiled_contexts] == [
        rank % 4,
        rank // 4,
        rank % 4,
    ]
    assert [item[1].world_size for item in compiled_contexts] == [4, 2, 4]


def test_compile_rejects_group_members_that_do_not_match_stage_topology() -> None:
    groups = {
        "intra_reduce_scatter": FakeGroup((0, 1, 2, 4)),
        "inter_ring": FakeGroup((0, 4)),
        "restore_full": FakeGroup((0, 1, 2, 3)),
    }

    with pytest.raises(ValueError, match="intra_reduce_scatter.*expected.*0, 1, 2, 3"):
        compile_hierarchical_stages(
            _plan(groups),
            _context(0),
            operation_factory=lambda stage, context: lambda value: value,
            group_members=lambda group: group.ranks,
        )


def test_compile_rejects_layout_chain_or_final_layout_mismatch() -> None:
    local = FakeGroup((0, 1, 2, 3))
    inter = FakeGroup((0, 4))
    groups = {
        "intra_reduce_scatter": local,
        "inter_ring": inter,
        "restore_full": local,
    }
    invalid = replace(_plan(groups), output_layout="shard")

    with pytest.raises(ValueError, match="final output layout"):
        compile_hierarchical_stages(
            invalid,
            _context(0),
            operation_factory=lambda stage, context: lambda value: value,
            group_members=lambda group: group.ranks,
        )


def test_group_factory_runs_once_for_reused_local_group() -> None:
    plan = _plan(
        {
            "intra_reduce_scatter": None,
            "inter_ring": None,
            "restore_full": None,
        }
    )
    created: list[tuple[int, ...]] = []

    def group_factory(ranks: tuple[int, ...]):
        created.append(ranks)
        return FakeGroup(ranks)

    executor = compile_hierarchical_stages(
        plan,
        _context(0),
        operation_factory=lambda stage, context: lambda value: value,
        group_factory=group_factory,
        group_members=lambda group: group.ranks,
    )

    assert created == [(0, 1, 2, 3), (0, 4)]
    assert executor.stages[0].process_group is executor.stages[2].process_group


def test_executor_orders_stages_with_stream_events_without_host_waits() -> None:
    calls: list[tuple[object, ...]] = []

    def stage(name: str, input_layout: str, output_layout: str) -> CompiledStage:
        stream = f"{name}_stream"

        def operation(value):
            calls.append(("run", name, value))
            result = f"{value}:{name}"
            return StageExecution(
                result,
                completion=FakeCompletion(name, calls),
                resources=(result,),
            )

        return CompiledStage(
            name=name,
            input_layout=input_layout,
            output_layout=output_layout,
            participants=(0, 1),
            process_group=object(),
            stream=stream,
            operation=operation,
        )

    executor = HierarchicalExecutor(
        (
            stage("local", "full", "shard"),
            stage("inter", "shard", "shard"),
            stage("restore", "shard", "full"),
        )
    )

    work = executor.run("tensor")

    assert calls == [
        ("run", "local", "tensor"),
        ("wait_stream", "local", "inter_stream"),
        ("run", "inter", "tensor:local"),
        ("wait_stream", "inter", "restore_stream"),
        ("run", "restore", "tensor:local:inter"),
    ]
    assert work.query() is True
    assert work.wait() == "tensor:local:inter:restore"
    assert calls[-1] == ("wait", "restore")
    assert len(work.resources) == 3
