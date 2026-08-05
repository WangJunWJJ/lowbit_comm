from __future__ import annotations

from dataclasses import replace
import gc
import weakref

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


def test_compile_rejects_final_layout_mismatch_before_stage_compilation() -> None:
    local = FakeGroup((0, 1, 2, 3))
    inter = FakeGroup((0, 4))
    groups = {
        "intra_reduce_scatter": local,
        "inter_ring": inter,
        "restore_full": local,
    }
    invalid = replace(_plan(groups), output_layout="shard")

    with pytest.raises(ValueError, match="canonical three-stage chain"):
        compile_hierarchical_stages(
            invalid,
            _context(0),
            operation_factory=lambda stage, context: lambda value: value,
            group_members=lambda group: group.ranks,
        )


@pytest.mark.parametrize(
    "stages",
    [
        lambda valid: (valid[0], valid[2]),
        lambda valid: (*valid, valid[2]),
        lambda valid: (valid[0], replace(valid[1], strategy="all_gather"), valid[2]),
    ],
    ids=("missing_inter", "extra_restore", "wrong_inter_strategy"),
)
def test_compile_rejects_noncanonical_stage_chain_before_compiling_operations(
    stages,
) -> None:
    local = FakeGroup((0, 1, 2, 3))
    inter = FakeGroup((0, 4))
    groups = {
        "intra_reduce_scatter": local,
        "inter_ring": inter,
        "restore_full": local,
    }
    valid = _plan(groups)
    invalid = replace(valid, stages=stages(valid.stages))
    compiled_operations = []

    with pytest.raises(ValueError, match="canonical three-stage chain"):
        compile_hierarchical_stages(
            invalid,
            _context(0),
            operation_factory=lambda stage, context: compiled_operations.append(stage),
            group_members=lambda group: group.ranks,
        )

    assert compiled_operations == []


def test_group_factory_uses_one_global_deterministic_creation_order() -> None:
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

    assert created == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
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
    assert work.resources[0] == "tensor"
    assert len(work.resources) == 4


def test_executor_forwards_caller_output_only_to_final_stage() -> None:
    calls = []
    output = object()

    def passthrough(name, input_layout, output_layout):
        def operation(value):
            calls.append((name, value, None))
            return value

        return CompiledStage(
            name=name,
            input_layout=input_layout,
            output_layout=output_layout,
            participants=(0, 1),
            process_group=object(),
            operation=operation,
        )

    def restore(value, *, out):
        calls.append(("restore", value, out))
        return out

    executor = HierarchicalExecutor(
        (
            passthrough("local", "full", "shard"),
            passthrough("inter", "shard", "shard"),
            CompiledStage(
                name="restore",
                input_layout="shard",
                output_layout="full",
                participants=(0, 1),
                process_group=object(),
                operation=restore,
            ),
        )
    )

    work = executor.run("tensor", out=output)

    assert work.wait() is output
    assert calls == [
        ("local", "tensor", None),
        ("inter", "tensor", None),
        ("restore", "tensor", output),
    ]


def test_executor_quarantines_submitted_resources_when_later_stage_fails() -> None:
    class Resource:
        pass

    class ToggleCompletion:
        def __init__(self) -> None:
            self.ready = False

        def wait_stream(self, stream) -> None:
            pass

        def query(self) -> bool:
            return self.ready

        def wait(self) -> None:
            self.ready = True

    producer = ToggleCompletion()
    emergency = ToggleCompletion()
    resource_ref = None

    def produce(value):
        nonlocal resource_ref
        resource = Resource()
        resource_ref = weakref.ref(resource)
        return StageExecution(
            "shard",
            completion=producer,
            resources=(resource,),
        )

    def fail(value):
        raise RuntimeError("stage failed after enqueue")

    executor = HierarchicalExecutor(
        (
            CompiledStage(
                "local",
                "full",
                "shard",
                (0, 1),
                object(),
                produce,
                stream="local_stream",
            ),
            CompiledStage(
                "inter",
                "shard",
                "shard",
                (0, 2),
                object(),
                fail,
                stream="inter_stream",
                completion_factory=lambda value, stream: emergency,
            ),
        )
    )

    with pytest.raises(RuntimeError, match="stage failed after enqueue"):
        executor.run("tensor")

    gc.collect()
    assert executor.pending_failure_count == 1
    assert resource_ref is not None and resource_ref() is not None
    assert executor.reap_pending_failures() == 0

    emergency.ready = True
    assert executor.reap_pending_failures() == 1
    gc.collect()
    assert resource_ref() is None
