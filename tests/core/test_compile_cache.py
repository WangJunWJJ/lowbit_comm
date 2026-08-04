from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Lock
from time import sleep

import pytest

from ccdl_comm import (
    BackendCapabilities,
    BackendKey,
    BackendRegistry,
    CommunicationPlan,
    CompileContext,
    CompressionConfig,
    ExecutionInfo,
    ImmediateWork,
    WorkspacePolicy,
)
from ccdl_comm.compiler import CompileCache, compile as compile_plan


BASE_CONTEXT = CompileContext(
    rank=0,
    world_size=2,
    device="cuda:0",
    shape=(1024,),
    dtype="float16",
    process_group=object(),
    topology_signature="pcie-single-node",
)
BASE_PLAN = CommunicationPlan(
    "all_reduce",
    "ring",
    backend="fake",
    compression=CompressionConfig(bit=8, group_size=64),
)


@dataclass
class FakeExecutor:
    execution_info: ExecutionInfo

    def run(self, tensor: object) -> ImmediateWork[object]:
        return ImmediateWork(tensor)


class CountingBackend:
    name = "fake"
    abi_version = 1

    def __init__(self) -> None:
        self.compile_calls = 0

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            available=True,
            collectives={"all_reduce"},
            strategies={"ring"},
            dtypes={"float16", "bfloat16"},
            bits={4, 8},
            output_layouts={"full"},
            supports_async=True,
        )

    def compile(self, plan: CommunicationPlan, context: CompileContext) -> FakeExecutor:
        self.compile_calls += 1
        return FakeExecutor(
            ExecutionInfo(
                requested_strategy=plan.strategy,
                executed_strategy=plan.strategy,
                backend=plan.backend,
                fallback_used=False,
                fallback_reason=None,
                stage_names=(),
                original_bytes=0,
                compressed_bytes=0,
                compression_ratio=1.0,
                workspace_cache_hit=False,
                async_capable=plan.async_op,
                fast_path="fake",
            )
        )


def _registry(backend: CountingBackend) -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(BackendKey("all_reduce", "ring", "fake", "full"), lambda: backend)
    return registry


def test_compile_cache_reuses_same_exact_shape_class() -> None:
    backend = CountingBackend()
    registry = _registry(backend)
    cache = CompileCache(max_entries=4)

    first = compile_plan(BASE_PLAN, BASE_CONTEXT, registry=registry, cache=cache)
    second = compile_plan(BASE_PLAN, BASE_CONTEXT, registry=registry, cache=cache)

    assert second is first
    assert backend.compile_calls == 1
    assert len(cache) == 1


@pytest.mark.parametrize(
    ("plan", "context"),
    [
        (BASE_PLAN, replace(BASE_CONTEXT, shape=(2048,))),
        (BASE_PLAN, replace(BASE_CONTEXT, dtype="bfloat16")),
        (BASE_PLAN, replace(BASE_CONTEXT, world_size=4)),
        (BASE_PLAN, replace(BASE_CONTEXT, process_group=object())),
        (
            replace(
                BASE_PLAN,
                compression=CompressionConfig(bit=4, group_size=64, allow_experimental=True),
            ),
            BASE_CONTEXT,
        ),
        (replace(BASE_PLAN, compression=CompressionConfig(bit=8, group_size=32)), BASE_CONTEXT),
        (BASE_PLAN, replace(BASE_CONTEXT, layout="channels_last")),
        (BASE_PLAN, replace(BASE_CONTEXT, topology_signature="nvlink-single-node")),
        (
            replace(BASE_PLAN, workspace_policy=WorkspacePolicy(max_entries=2)),
            BASE_CONTEXT,
        ),
    ],
    ids=[
        "shape",
        "dtype",
        "world-size",
        "process-group-identity",
        "bit",
        "group-size",
        "layout",
        "topology-signature",
        "workspace-policy",
    ],
)
def test_compile_cache_key_changes_for_required_dimensions(
    plan: CommunicationPlan,
    context: CompileContext,
) -> None:
    backend = CountingBackend()
    registry = _registry(backend)
    cache = CompileCache(max_entries=4)

    baseline = compile_plan(BASE_PLAN, BASE_CONTEXT, registry=registry, cache=cache)
    changed = compile_plan(plan, context, registry=registry, cache=cache)

    assert changed is not baseline
    assert backend.compile_calls == 2


def test_process_group_cache_identity_does_not_use_equality_or_string_value() -> None:
    class EqualButDistinctGroup:
        __hash__ = None

        def __eq__(self, other: object) -> bool:
            return isinstance(other, EqualButDistinctGroup)

        def __str__(self) -> str:
            return "same-group"

    backend = CountingBackend()
    registry = _registry(backend)
    cache = CompileCache(max_entries=4)
    first_context = replace(BASE_CONTEXT, process_group=EqualButDistinctGroup())
    second_context = replace(BASE_CONTEXT, process_group=EqualButDistinctGroup())

    first = compile_plan(BASE_PLAN, first_context, registry=registry, cache=cache)
    second = compile_plan(BASE_PLAN, second_context, registry=registry, cache=cache)

    assert second is not first
    assert backend.compile_calls == 2


def test_compile_cache_is_bounded_lru() -> None:
    backend = CountingBackend()
    registry = _registry(backend)
    cache = CompileCache(max_entries=2)
    context_a = replace(BASE_CONTEXT, shape=(1,))
    context_b = replace(BASE_CONTEXT, shape=(2,))
    context_c = replace(BASE_CONTEXT, shape=(3,))

    compiled_a = compile_plan(BASE_PLAN, context_a, registry=registry, cache=cache)
    compile_plan(BASE_PLAN, context_b, registry=registry, cache=cache)
    assert compile_plan(BASE_PLAN, context_a, registry=registry, cache=cache) is compiled_a
    compile_plan(BASE_PLAN, context_c, registry=registry, cache=cache)
    compile_plan(BASE_PLAN, context_b, registry=registry, cache=cache)

    assert backend.compile_calls == 4
    assert len(cache) == 2


def test_compile_cache_serializes_same_key_creation() -> None:
    class SlowBackend(CountingBackend):
        def __init__(self) -> None:
            super().__init__()
            self._counter_lock = Lock()

        def compile(self, plan: CommunicationPlan, context: CompileContext) -> FakeExecutor:
            with self._counter_lock:
                self.compile_calls += 1
            sleep(0.05)
            return FakeExecutor(
                ExecutionInfo(
                    requested_strategy=plan.strategy,
                    executed_strategy=plan.strategy,
                    backend=plan.backend,
                    fallback_used=False,
                    fallback_reason=None,
                    stage_names=(),
                    original_bytes=0,
                    compressed_bytes=0,
                    compression_ratio=1.0,
                    workspace_cache_hit=False,
                    async_capable=True,
                    fast_path="fake",
                )
            )

    backend = SlowBackend()
    registry = _registry(backend)
    cache = CompileCache(max_entries=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                compile_plan,
                BASE_PLAN,
                BASE_CONTEXT,
                registry=registry,
                cache=cache,
            )
            for _ in range(2)
        ]
    compiled = [future.result() for future in futures]

    assert compiled[0] is compiled[1]
    assert backend.compile_calls == 1


def test_compile_without_cache_does_not_reuse_executor() -> None:
    backend = CountingBackend()
    registry = _registry(backend)

    first = compile_plan(BASE_PLAN, BASE_CONTEXT, registry=registry)
    second = compile_plan(BASE_PLAN, BASE_CONTEXT, registry=registry)

    assert second is not first
    assert backend.compile_calls == 2


def test_compile_cache_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        CompileCache(max_entries=0)
