from __future__ import annotations

from dataclasses import dataclass, replace

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
)
from ccdl_comm.compiler import compile as compile_plan
from ccdl_comm.exceptions import BackendRegistrationError, UnsupportedCollective


CONTEXT = CompileContext(
    rank=0,
    world_size=2,
    device="cuda:0",
    shape=(1024,),
    dtype="float16",
)


def _execution_info(plan: CommunicationPlan) -> ExecutionInfo:
    return ExecutionInfo(
        requested_strategy=plan.strategy,
        executed_strategy=plan.strategy,
        backend=plan.backend,
        fallback_used=False,
        fallback_reason=None,
        stage_names=tuple(stage.name for stage in plan.stages),
        original_bytes=2048,
        compressed_bytes=1024,
        compression_ratio=0.5,
        workspace_cache_hit=False,
        async_capable=plan.async_op,
        fast_path="fake",
    )


@dataclass
class FakeExecutor:
    backend: "FakeBackend"
    execution_info: ExecutionInfo

    def run(self, tensor: object) -> ImmediateWork[object]:
        self.backend.run_calls += 1
        return ImmediateWork(tensor)


class FakeBackend:
    name = "fake"
    abi_version = 1

    def __init__(
        self,
        *,
        available: bool = True,
        capabilities: BackendCapabilities | None = None,
    ) -> None:
        self.available = available
        self.declared_capabilities = capabilities
        self.capability_calls = 0
        self.compile_calls = 0
        self.run_calls = 0
        self.compiled_plans: list[CommunicationPlan] = []

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        self.capability_calls += 1
        if self.declared_capabilities is not None:
            return self.declared_capabilities
        return BackendCapabilities(
            backend=self.name,
            available=self.available,
            collectives={"all_reduce"},
            strategies={"ring", "all_gather", "reduce_scatter"},
            dtypes={"float16", "bfloat16"},
            output_layouts={"full"},
            supports_async=True,
            supports_dynamic_shape=True,
            reason=None if self.available else "fake backend disabled",
        )

    def compile(self, plan: CommunicationPlan, context: CompileContext) -> FakeExecutor:
        self.compile_calls += 1
        self.compiled_plans.append(plan)
        return FakeExecutor(self, _execution_info(plan))


@dataclass
class FakeReducedShardExecutor:
    execution_info: ExecutionInfo
    calls: list[tuple[object, object | None]]

    def run(self, tensor: object, *, out: object | None = None) -> ImmediateWork[object]:
        self.calls.append((tensor, out))
        return ImmediateWork(out if out is not None else tensor)


class FakeReducedShardBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.executor: FakeReducedShardExecutor | None = None

    def capabilities(self, context: CompileContext) -> BackendCapabilities:
        self.capability_calls += 1
        return BackendCapabilities(
            backend=self.name,
            available=True,
            collectives={"reduce_scatter"},
            strategies={"compressed"},
            dtypes={"float16"},
            bits={8},
            output_layouts={"shard"},
            supports_async=True,
        )

    def compile(self, plan: CommunicationPlan, context: CompileContext) -> FakeReducedShardExecutor:
        self.compile_calls += 1
        self.compiled_plans.append(plan)
        self.executor = FakeReducedShardExecutor(_execution_info(plan), [])
        return self.executor


def _registry(backend: FakeBackend, strategy: str = "ring") -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        BackendKey("all_reduce", strategy, "fake", "full"),
        lambda: backend,
    )
    return registry


def test_compiled_plan_does_not_resolve_backend_on_run() -> None:
    backend = FakeBackend()
    compiled = compile_plan(
        CommunicationPlan("all_reduce", "ring", backend="fake"),
        CONTEXT,
        registry=_registry(backend),
    )

    assert compiled.run("a").wait() == "a"
    assert compiled.run("b").wait() == "b"
    assert backend.capability_calls == 1
    assert backend.compile_calls == 1
    assert backend.run_calls == 2


def test_compiled_reduced_shard_plan_forwards_caller_owned_output() -> None:
    backend = FakeReducedShardBackend()
    registry = BackendRegistry()
    registry.register(
        BackendKey("reduce_scatter", "compressed", "fake", "shard"),
        lambda: backend,
    )
    compiled = compile_plan(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            backend="fake",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
        ),
        CONTEXT,
        registry=registry,
    )
    output = object()

    assert compiled.run("bucket", out=output).wait() is output
    assert backend.executor is not None
    assert backend.executor.calls == [("bucket", output)]


def test_compiled_non_shard_plan_rejects_caller_owned_output() -> None:
    backend = FakeBackend()
    compiled = compile_plan(
        CommunicationPlan("all_reduce", "ring", backend="fake"),
        CONTEXT,
        registry=_registry(backend),
    )

    with pytest.raises(TypeError, match="caller-owned shard output"):
        compiled.run("bucket", out=object())


def test_compile_passes_effective_fallback_plan_to_backend() -> None:
    backend = FakeBackend()
    plan = CommunicationPlan(
        "all_reduce",
        "ring",
        backend="fake",
        fallback=("all_gather",),
    )

    compiled = compile_plan(plan, CONTEXT, registry=_registry(backend, "all_gather"))

    assert backend.compiled_plans[0].strategy == "all_gather"
    assert compiled.execution_info.requested_strategy == "ring"
    assert compiled.execution_info.executed_strategy == "all_gather"
    assert compiled.execution_info.fallback_used is True
    assert compiled.execution_info.fallback_reason == "explicit fallback from ring to all_gather"
    assert compiled.executor.execution_info is compiled.execution_info


def test_compile_uses_fallback_when_registered_strategy_is_contextually_unavailable() -> None:
    unavailable = FakeBackend(available=False)
    fallback = FakeBackend()
    registry = BackendRegistry()
    registry.register(BackendKey("all_reduce", "ring", "fake", "full"), lambda: unavailable)
    registry.register(BackendKey("all_reduce", "all_gather", "fake", "full"), lambda: fallback)

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "ring",
            backend="fake",
            fallback=("all_gather",),
        ),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "all_gather"
    assert compiled.execution_info.fallback_used is True
    assert unavailable.compile_calls == 0
    assert fallback.compile_calls == 1


def test_auto_skips_contextually_unavailable_preferred_strategy() -> None:
    unavailable = FakeBackend(available=False)
    fallback = FakeBackend()
    registry = BackendRegistry()
    registry.register(
        BackendKey("all_reduce", "reduce_scatter", "fake", "full"),
        lambda: unavailable,
    )
    registry.register(BackendKey("all_reduce", "all_gather", "fake", "full"), lambda: fallback)

    compiled = compile_plan(
        CommunicationPlan("all_reduce", "auto", backend="fake"),
        replace(CONTEXT, world_size=4),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "all_gather"
    assert compiled.execution_info.fallback_used is True
    assert "reduce_scatter" in compiled.execution_info.fallback_reason


def test_auto_never_compiles_an_auto_backend_strategy() -> None:
    auto_backend = FakeBackend(
        capabilities=BackendCapabilities(
            backend="fake",
            available=True,
            collectives={"all_reduce"},
            strategies={"auto"},
            dtypes={"float16"},
            output_layouts={"full"},
            supports_async=True,
        )
    )
    concrete_backend = FakeBackend()
    registry = BackendRegistry()
    registry.register(BackendKey("all_reduce", "auto", "fake", "full"), lambda: auto_backend)
    registry.register(
        BackendKey("all_reduce", "all_gather", "fake", "full"),
        lambda: concrete_backend,
    )

    compiled = compile_plan(
        CommunicationPlan("all_reduce", "auto", backend="fake"),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.requested_strategy == "auto"
    assert compiled.execution_info.executed_strategy == "all_gather"
    assert auto_backend.compile_calls == 0
    assert concrete_backend.compile_calls == 1


def test_explicit_fallback_never_compiles_an_auto_backend_strategy() -> None:
    auto_backend = FakeBackend(
        capabilities=BackendCapabilities(
            backend="fake",
            available=True,
            collectives={"all_reduce"},
            strategies={"auto"},
            dtypes={"float16"},
            output_layouts={"full"},
            supports_async=True,
        )
    )
    concrete_backend = FakeBackend()
    registry = BackendRegistry()
    registry.register(BackendKey("all_reduce", "auto", "fake", "full"), lambda: auto_backend)
    registry.register(
        BackendKey("all_reduce", "all_gather", "fake", "full"),
        lambda: concrete_backend,
    )

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "ring",
            backend="fake",
            fallback=("auto", "all_gather"),
        ),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "all_gather"
    assert auto_backend.compile_calls == 0
    assert concrete_backend.compile_calls == 1


def test_compile_rejects_unavailable_backend_before_backend_compile() -> None:
    backend = FakeBackend(available=False)

    with pytest.raises(UnsupportedCollective, match="fake backend disabled"):
        compile_plan(
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            CONTEXT,
            registry=_registry(backend),
        )

    assert backend.capability_calls == 1
    assert backend.compile_calls == 0


@pytest.mark.parametrize(
    ("capabilities", "context", "plan", "message"),
    [
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"reduce_scatter"},
                strategies={"ring"},
                dtypes={"float16"},
                output_layouts={"full"},
                supports_async=True,
            ),
            CONTEXT,
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "collective",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"tree"},
                dtypes={"float16"},
                output_layouts={"full"},
                supports_async=True,
            ),
            CONTEXT,
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "strategy",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"ring"},
                dtypes={"bfloat16"},
                output_layouts={"full"},
                supports_async=True,
            ),
            CONTEXT,
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "dtype",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"ring"},
                dtypes={"float16"},
                bits={4},
                output_layouts={"full"},
                supports_async=True,
            ),
            CONTEXT,
            CommunicationPlan(
                "all_reduce",
                "ring",
                backend="fake",
                compression=CompressionConfig(bit=8),
            ),
            "bit",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"ring"},
                dtypes={"float16"},
                output_layouts={"shard"},
                supports_async=True,
            ),
            CONTEXT,
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "output layout",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"ring"},
                dtypes={"float16"},
                output_layouts={"full"},
                supports_async=False,
            ),
            CONTEXT,
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "async",
        ),
        (
            BackendCapabilities(
                backend="fake",
                available=True,
                collectives={"all_reduce"},
                strategies={"ring"},
                dtypes={"float16"},
                output_layouts={"full"},
                supports_async=True,
                supports_dynamic_shape=False,
            ),
            replace(CONTEXT, allow_dynamic_shape=True),
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            "dynamic shape",
        ),
    ],
)
def test_compile_rejects_unsupported_capability_dimensions(
    capabilities: BackendCapabilities,
    context: CompileContext,
    plan: CommunicationPlan,
    message: str,
) -> None:
    backend = FakeBackend(capabilities=capabilities)

    with pytest.raises(UnsupportedCollective, match=message):
        compile_plan(plan, context, registry=_registry(backend))

    assert backend.compile_calls == 0


def test_compile_rejects_backend_executor_contract_violation() -> None:
    class InvalidBackend(FakeBackend):
        def compile(self, plan: CommunicationPlan, context: CompileContext) -> object:
            return object()

    backend = InvalidBackend()

    with pytest.raises(BackendRegistrationError, match="CompiledExecutor"):
        compile_plan(
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            CONTEXT,
            registry=_registry(backend),
        )


def test_compile_rejects_backend_registered_under_wrong_backend_name() -> None:
    backend = FakeBackend()
    backend.name = "different"

    with pytest.raises(BackendRegistrationError, match="backend name"):
        compile_plan(
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            CONTEXT,
            registry=_registry(backend),
        )


def test_compile_rejects_inconsistent_executor_execution_info() -> None:
    class InconsistentBackend(FakeBackend):
        def compile(self, plan: CommunicationPlan, context: CompileContext) -> FakeExecutor:
            executor = super().compile(plan, context)
            executor.execution_info = _execution_info(
                CommunicationPlan("all_reduce", "tree", backend="fake")
            )
            return executor

    backend = InconsistentBackend()

    with pytest.raises(BackendRegistrationError, match="executed strategy"):
        compile_plan(
            CommunicationPlan("all_reduce", "ring", backend="fake"),
            CONTEXT,
            registry=_registry(backend),
        )


def test_compile_requires_registered_backend_when_registry_is_omitted() -> None:
    with pytest.raises(UnsupportedCollective, match="all_reduce:ring"):
        compile_plan(
            CommunicationPlan("all_reduce", "ring", backend="unregistered"),
            CONTEXT,
        )


def test_compilation_api_is_exported_from_package_root() -> None:
    import ccdl_comm

    assert ccdl_comm.compile is compile_plan
    assert ccdl_comm.CompileCache.__name__ == "CompileCache"
    assert ccdl_comm.CompiledCommunicationPlan.__name__ == "CompiledCommunicationPlan"
