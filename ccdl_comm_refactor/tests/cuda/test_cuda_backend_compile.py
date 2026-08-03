from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from dataclasses import replace

import pytest

from ccdl_comm import (
    CommunicationPlan,
    CommunicationStage,
    CompileContext,
    CompressionConfig,
    ImmediateWork,
    ReducedShard,
    WorkspacePolicy,
    compile as compile_plan,
)
from ccdl_comm.cuda.backend import CudaCommunicationBackend
from ccdl_comm.cuda.backend import register_cuda_backends
from ccdl_comm.cuda.executors import (
    CompressedReduceScatterExecutor,
    CudaAllReduceExecutor,
    CudaReducedShardExecutor,
)
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.exceptions import UnsupportedCollective
from ccdl_comm.registry import BackendKey, BackendRegistry


CONTEXT = CompileContext(
    rank=0,
    world_size=2,
    device="cuda:0",
    shape=(1024,),
    dtype="float16",
    topology_signature="pcie-single-node",
)
EXTENSION = CudaExtensionStatus(
    available=True,
    module=type(
        "NativeExtension",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(lambda: object()),
            "inplace_dequantize_reduce_mean": staticmethod(lambda *args: True),
        },
    )(),
)


def test_cuda_reduced_shard_compiler_adapts_transport_callback_to_codec_facade(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    captured = {}
    facade_calls = []

    def make_transport(**kwargs):
        captured.update(kwargs)
        return lambda tensor, **operation_kwargs: tensor

    def facade(buffers, output, config, *, extension_status, reduce):
        facade_calls.append((buffers, output, config, extension_status, reduce))
        return True

    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    monkeypatch.setattr(compiler_module, "inplace_dequantize_reduce_mean", facade)
    CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
        ),
        CONTEXT,
    )

    output = object()
    assert captured["fused_dequantize_reduce"](["payload"], output, reduce="mean") is True
    assert facade_calls == [(["payload"], output, CompressionConfig(bit=8), EXTENSION, "mean")]


@pytest.mark.parametrize(
    ("config", "context", "status", "expected_reason"),
    [
        (CompressionConfig(bit=4, allow_experimental=True), CONTEXT, EXTENSION, "bit=8"),
        (CompressionConfig(bit=8), replace(CONTEXT, world_size=9), EXTENSION, "at most 8"),
        (CompressionConfig(bit=8), CONTEXT, CudaExtensionStatus(False, None, "not built"), "not built"),
        (CompressionConfig(bit=8), CONTEXT, CudaExtensionStatus(True, object()), "does not export"),
    ],
)
def test_cuda_reduced_shard_compiler_uses_one_capability_result_for_callback_and_execution_info(
    monkeypatch,
    config,
    context,
    status,
    expected_reason,
) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    captured = {}

    def make_transport(**kwargs):
        captured.update(kwargs)
        return lambda tensor, **operation_kwargs: tensor

    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    executor = compiler_module.compile_cuda_plan(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=config,
            output_layout="shard",
            async_op=False,
        ),
        context,
        status,
    )

    assert captured["fused_dequantize_reduce"] is None
    assert expected_reason in captured["fused_dequantize_reduce_reason"]
    assert executor.execution_info.details["cuda_fused_reduced_shard"] is False
    assert executor.execution_info.details["cuda_fused_reduced_shard_fallback_reason"] == captured[
        "fused_dequantize_reduce_reason"
    ]


def test_cuda_reduced_shard_capability_rejects_unsupported_dtype() -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    capability = compiler_module._fused_reduced_shard_capability(
        CompressionConfig(bit=8),
        dtype="float64",
        world_size=2,
        extension_status=EXTENSION,
    )

    assert capability.available is False
    assert "dtype" in capability.reason


def test_cuda_backend_compiles_supported_int8_all_reduce() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)

    executor = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
        ),
        CONTEXT,
    )

    assert isinstance(executor, CudaAllReduceExecutor)
    assert executor.execution_info.backend == "cuda"
    assert executor.execution_info.executed_strategy == "all_gather"
    assert executor.execution_info.fast_path == "cuda_all_gather"


def test_native_nccl_backend_is_available_without_cuda_extension(monkeypatch) -> None:
    import ccdl_comm.cuda.backend as backend_module

    class FakeDist:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_backend(group=None) -> str:
            return "nccl"

    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeDist)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "extension unavailable"),
    )

    backend = registry.resolve(
        BackendKey("all_reduce", "native_nccl", "cuda", "full")
    )
    capabilities = backend.capabilities(replace(CONTEXT, process_group=object()))

    assert capabilities.available is True
    assert capabilities.strategies == {"native_nccl"}
    assert "native_nccl" in capabilities.features
    assert "cuda_extension" not in capabilities.features


def test_native_nccl_backend_rejects_non_nccl_process_group(monkeypatch) -> None:
    import ccdl_comm.cuda.backend as backend_module

    class FakeDist:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_backend(group=None) -> str:
            return "gloo"

    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeDist)

    with pytest.raises(UnsupportedCollective, match="NCCL"):
        CudaCommunicationBackend(
            extension_status=CudaExtensionStatus(False, None, "extension unavailable")
        ).compile(
            CommunicationPlan("all_reduce", "native_nccl", async_op=False),
            CONTEXT,
        )


def test_cuda_auto_large_full_falls_back_to_native_when_extension_is_missing(
    monkeypatch,
) -> None:
    import ccdl_comm.cuda.backend as backend_module
    import ccdl_comm.cuda.compiler as compiler_module

    class FakeDist:
        class ReduceOp:
            AVG = "avg"

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_backend(group=None) -> str:
            return "nccl"

        @staticmethod
        def all_reduce(tensor, **kwargs):
            return None

    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeDist)
    monkeypatch.setattr(compiler_module, "import_module", lambda name: FakeDist)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "extension unavailable"),
    )

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "auto",
            compression=CompressionConfig(),
            async_op=False,
        ),
        replace(
            CONTEXT,
            shape=(8_388_608,),
            device_architecture="NVIDIA RTX A6000",
        ),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "native_nccl"
    assert compiled.execution_info.fallback_used is True
    assert "topology" in (compiled.execution_info.fallback_reason or "")
    assert "extension unavailable" in (compiled.execution_info.fallback_reason or "")


@pytest.mark.parametrize("async_op", [False, True])
def test_native_nccl_executor_preserves_mean_and_work_semantics(
    monkeypatch,
    async_op: bool,
) -> None:
    import ccdl_comm.cuda.backend as backend_module
    import ccdl_comm.cuda.compiler as compiler_module

    calls = []

    class FakeHandle:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self) -> None:
            self.wait_calls += 1

        def is_completed(self) -> bool:
            return self.wait_calls > 0

    class FakeDist:
        class ReduceOp:
            AVG = "avg"

        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_backend(group=None) -> str:
            return "nccl"

        @staticmethod
        def all_reduce(tensor, *, op, group, async_op):
            calls.append((tensor, op, group, async_op))
            return FakeHandle() if async_op else None

    monkeypatch.setattr(
        compiler_module,
        "import_module",
        lambda name: FakeDist if name == "torch.distributed" else None,
    )
    monkeypatch.setattr(
        backend_module,
        "import_module",
        lambda name: FakeDist if name == "torch.distributed" else None,
    )
    process_group = object()
    executor = CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(False, None, "extension unavailable")
    ).compile(
        CommunicationPlan(
            "all_reduce",
            "native_nccl",
            compression=None,
            async_op=async_op,
        ),
        replace(CONTEXT, process_group=process_group),
    )
    tensor = object()

    work = executor.run(tensor)

    assert work.wait() is tensor
    assert calls == [(tensor, "avg", process_group, async_op)]
    assert executor.execution_info.fast_path == "cuda_native_nccl"
    assert executor.execution_info.compression_ratio == 1.0


@pytest.mark.parametrize("world_size", [2, 4])
def test_cuda_auto_compiles_validated_large_full_bucket_to_ring(
    world_size: int,
) -> None:
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=EXTENSION)

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "auto",
            compression=CompressionConfig(),
            output_layout="full",
            async_op=False,
        ),
        replace(
            CONTEXT,
            world_size=world_size,
            shape=(8_388_608,),
            device_architecture="NVIDIA RTX A6000",
        ),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "topology"
    assert compiled.execution_info.fallback_used is False
    assert compiled.execution_info.details["strategy_benchmark_matched"] is True
    assert compiled.executor._operation.topology_method == "ring"  # noqa: SLF001


@pytest.mark.parametrize("world_size", [2, 4])
def test_cuda_auto_compiles_validated_large_shard_without_fallback(
    world_size: int,
) -> None:
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=EXTENSION)

    compiled = compile_plan(
        CommunicationPlan(
            "reduce_scatter",
            "auto",
            compression=CompressionConfig(),
            output_layout="shard",
            async_op=False,
        ),
        replace(
            CONTEXT,
            world_size=world_size,
            shape=(8_388_608,),
            device_architecture="NVIDIA RTX A6000",
        ),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "compressed"
    assert compiled.execution_info.fallback_used is False
    assert compiled.execution_info.details["strategy_benchmark_matched"] is True


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 8])
def test_cuda_auto_unknown_or_small_full_bucket_compiles_native_without_fallback(
    monkeypatch,
    world_size: int,
) -> None:
    import ccdl_comm.cuda.backend as backend_module
    import ccdl_comm.cuda.compiler as compiler_module

    class FakeDist:
        class ReduceOp:
            AVG = "avg"

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_backend(group=None):
            return "nccl"

        @staticmethod
        def all_reduce(tensor, **kwargs):
            return None

    monkeypatch.setattr(compiler_module, "import_module", lambda name: FakeDist)
    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeDist)
    registry = BackendRegistry()
    register_cuda_backends(
        registry,
        extension_status=CudaExtensionStatus(False, None, "extension unavailable"),
    )

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "auto",
            compression=CompressionConfig(),
            output_layout="full",
            async_op=False,
        ),
        replace(
            CONTEXT,
            world_size=world_size,
            shape=(524_288,),
            device_architecture="unknown",
        ),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "native_nccl"
    assert compiled.execution_info.fallback_used is False
    assert compiled.execution_info.details["strategy_benchmark_matched"] is False


def test_cuda_backend_compiles_topology_and_hierarchical_stage_plan(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    class FakeGroup:
        def __init__(self, ranks):
            self.ranks = tuple(ranks)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def Stream(device=None):
            return object()

    class FakeTorch:
        cuda = FakeCuda()

    class FakeDist:
        @staticmethod
        def get_process_group_ranks(group):
            return group.ranks

    monkeypatch.setattr(
        compiler_module,
        "import_module",
        lambda name: FakeTorch if name == "torch" else FakeDist,
    )
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    topology = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=False,
        ),
        CONTEXT,
    )
    assert isinstance(topology, CudaAllReduceExecutor)
    assert topology.execution_info.fast_path == "cuda_topology"
    assert topology.execution_info.async_capable is True
    assert topology._operation.topology_method == "ring"  # noqa: SLF001
    local_group = FakeGroup((0, 1))
    inter_group = FakeGroup((0,))
    hierarchical = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "hierarchical",
            compression=CompressionConfig(bit=8),
            stages=(
                CommunicationStage(
                    "local",
                    "reduce_scatter",
                    "compressed",
                    compression=CompressionConfig(bit=8),
                    process_group=local_group,
                    output_layout="shard",
                    async_op=False,
                ),
                CommunicationStage(
                    "inter",
                    "all_reduce",
                    "topology",
                    compression=CompressionConfig(bit=8),
                    process_group=inter_group,
                    output_layout="shard",
                    async_op=False,
                ),
                CommunicationStage(
                    "restore",
                    "all_gather",
                    "native_nccl",
                    process_group=local_group,
                    output_layout="full",
                    async_op=False,
                ),
            ),
            async_op=False,
        ),
        replace(
            CONTEXT,
            local_rank=0,
            local_world_size=2,
            node_id=0,
            node_count=1,
        ),
    )

    assert isinstance(hierarchical, CudaAllReduceExecutor)
    assert hierarchical.execution_info.fast_path == "cuda_hierarchical"
    assert hierarchical.execution_info.details["hierarchical_recommended"] is False
    assert "single-node" in hierarchical.execution_info.details[
        "hierarchical_recommendation_reason"
    ]
    stage_executor = hierarchical._operation.hierarchical_executor  # noqa: SLF001
    assert tuple(stage.name for stage in stage_executor.stages) == (
        "local",
        "inter",
        "restore",
    )
    assert len({id(stage.stream) for stage in stage_executor.stages}) == 3


def test_cuda_backend_compiles_divisible_four_rank_topology_to_ring() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)

    topology = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=True,
        ),
        replace(CONTEXT, rank=2, world_size=4, shape=(4096,)),
    )

    assert topology._operation.topology_method == "ring"  # noqa: SLF001
    assert topology._operation.chunk_plan.world_size == 4  # noqa: SLF001


def test_cuda_backend_uses_tree_when_ring_shards_are_not_group_aligned() -> None:
    topology = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8, group_size=64),
            async_op=True,
        ),
        replace(CONTEXT, rank=2, world_size=4, shape=(128,)),
    )

    assert topology._operation.topology_method == "tree"  # noqa: SLF001


def test_cuda_backend_rejects_unsafe_unaligned_topology_output() -> None:
    with pytest.raises(UnsupportedCollective, match="group-aligned"):
        CudaCommunicationBackend(extension_status=EXTENSION).compile(
            CommunicationPlan(
                "all_reduce",
                "topology",
                compression=CompressionConfig(bit=8, group_size=64),
                async_op=True,
            ),
            replace(CONTEXT, shape=(4097,)),
        )


def test_cuda_backend_compiles_topology_with_workspace_cache_disabled() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)

    topology = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=True,
            workspace_policy=WorkspacePolicy(cache=False),
        ),
        CONTEXT,
    )

    assert topology.workspace_pool is not None


def test_cuda_backend_compiles_reduced_shard_executor() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)

    executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
        ),
        CONTEXT,
    )

    assert isinstance(executor, CudaReducedShardExecutor)
    assert isinstance(executor, CompressedReduceScatterExecutor)
    assert executor.chunk_plan.original_numel == 1024
    assert executor.chunk_plan.world_size == 2
    assert executor.execution_info.executed_strategy == "compressed"
    assert executor.execution_info.fast_path == "cuda_reduced_shard"


def test_cuda_backend_binds_operation_once_and_executor_runs_many_times() -> None:
    builder_calls: list[tuple[str, tuple[int, ...]]] = []
    operation_calls: list[object] = []

    def operation_factory(plan, context, extension_status) -> Callable[[object], object]:
        builder_calls.append((plan.strategy, context.shape))

        def operation(tensor: object) -> object:
            operation_calls.append(tensor)
            return tensor

        return operation

    backend = CudaCommunicationBackend(
        extension_status=EXTENSION,
        operation_factories={("all_reduce", "all_gather", "full"): operation_factory},
    )
    executor = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
        ),
        CONTEXT,
    )

    first = executor.run("a")
    second = executor.run("b")

    assert isinstance(first, ImmediateWork)
    assert first.wait() == "a"
    assert second.wait() == "b"
    assert builder_calls == [("all_gather", (1024,))]
    assert operation_calls == ["a", "b"]


def test_reduced_shard_executor_preserves_reduced_shard_result() -> None:
    shard = ReducedShard(
        shard="local",
        shard_index=0,
        shard_numel=512,
        original_shape=(1024,),
        original_numel=1024,
        world_size=2,
        reduce="mean",
    )
    backend = CudaCommunicationBackend(
        extension_status=EXTENSION,
        operation_factories={
            ("reduce_scatter", "compressed", "shard"): lambda plan, context, status: lambda tensor: shard
        },
    )
    executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
        ),
        CONTEXT,
    )

    assert executor.run("tensor").wait() is shard


def test_reduced_shard_executor_passes_caller_owned_output_to_prebound_operation() -> None:
    output = object()
    calls = []
    shard = ReducedShard(
        shard=output,
        shard_index=0,
        shard_numel=512,
        original_shape=(1024,),
        original_numel=1024,
        world_size=2,
        reduce="mean",
        metadata={"output_ownership": "caller", "fused_dequant_reduce": True},
    )

    def operation(tensor, *, out=None):
        calls.append((tensor, out))
        return shard

    backend = CudaCommunicationBackend(
        extension_status=EXTENSION,
        operation_factories={
            ("reduce_scatter", "compressed", "shard"): lambda plan, context, status: operation
        },
    )
    executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
        ),
        CONTEXT,
    )

    reduced = executor.run("tensor", out=output).wait()

    assert reduced.shard is output
    assert reduced.metadata["output_ownership"] == "caller"
    assert reduced.metadata["fused_dequant_reduce"] is True
    assert calls == [("tensor", output)]


def test_cuda_backend_rejects_unsupported_plan_and_missing_extension() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    with pytest.raises(UnsupportedCollective, match="all_reduce:ring"):
        backend.compile(
            CommunicationPlan("all_reduce", "ring", compression=CompressionConfig(bit=8)),
            CONTEXT,
        )


def test_cuda_compiler_marks_python_work_fallback_for_legacy_extension() -> None:
    backend = CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(True, object()),
        operation_factories={
            ("all_reduce", "all_gather", "full"): lambda plan, context, status: lambda tensor: tensor
        },
    )

    executor = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
        ),
        CONTEXT,
    )

    assert executor.execution_info.fast_path == "python_fallback"


def test_cuda_compiler_marks_fallback_when_native_executor_factory_fails() -> None:
    def fail_factory():
        raise RuntimeError("ABI load failed")

    module = type(
        "BrokenNativeExtension",
        (),
        {
            "CompressedWork": object,
            "NATIVE_WORK_ABI_VERSION": 1,
            "create_cuda_executor": staticmethod(fail_factory),
        },
    )()
    backend = CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(True, module),
        operation_factories={
            ("all_reduce", "all_gather", "full"): lambda plan, context, status: lambda tensor: tensor
        },
    )

    executor = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
        ),
        CONTEXT,
    )

    assert executor.execution_info.fast_path == "python_fallback"

    unavailable = CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(False, None, "extension missing")
    )
    assert unavailable.capabilities(CONTEXT).available is False
    with pytest.raises(UnsupportedCollective, match="extension missing"):
        unavailable.compile(
            CommunicationPlan(
                "all_reduce",
                "all_gather",
                compression=CompressionConfig(bit=8),
            ),
            CONTEXT,
        )


def test_cuda_backend_requires_compression_and_static_supported_context() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    with pytest.raises(UnsupportedCollective, match="compression"):
        backend.compile(CommunicationPlan("all_reduce", "all_gather"), CONTEXT)
    with pytest.raises(UnsupportedCollective, match="dynamic shape"):
        backend.compile(
            CommunicationPlan(
                "all_reduce",
                "all_gather",
                compression=CompressionConfig(bit=8),
            ),
            CompileContext(
                rank=0,
                world_size=2,
                device="cuda:0",
                shape=(1024,),
                dtype="float16",
                allow_dynamic_shape=True,
            ),
        )


def test_default_all_gather_executor_binds_compile_context_process_group(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    group = object()
    calls = []

    def fake_run(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return "reduced"

    monkeypatch.setattr(compiler_module, "_run_compressed_all_reduce", fake_run)
    executor = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
            async_op=False,
        ),
        replace(CONTEXT, process_group=group),
    )

    assert executor.run("tensor").wait() == "reduced"
    assert calls[0][1]["process_group"] is group


def test_cuda_compiler_binds_extension_status_to_runtime_completion_manager(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    manager = object()
    manager_statuses = []
    operation_calls = []

    def create_manager(*, extension_status):
        manager_statuses.append(extension_status)
        return manager

    def fake_run(tensor, **kwargs):
        operation_calls.append(kwargs)
        return tensor

    monkeypatch.setattr(compiler_module, "CudaCompletionManager", create_manager)
    monkeypatch.setattr(compiler_module, "_run_compressed_all_reduce", fake_run)
    executor = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
            async_op=False,
        ),
        CONTEXT,
    )

    executor.run("tensor").wait()

    assert manager_statuses == [EXTENSION]
    assert operation_calls[0]["completion_manager"] is manager


@pytest.mark.parametrize(
    ("collective", "strategy", "output_layout"),
    [
        ("all_reduce", "topology", "full"),
        ("reduce_scatter", "compressed", "shard"),
    ],
)
def test_cuda_backend_rejects_unsafe_subgroup_paths(
    collective,
    strategy,
    output_layout,
) -> None:
    plan = CommunicationPlan(
        collective,
        strategy,
        compression=CompressionConfig(bit=8),
        output_layout=output_layout,
        async_op=False,
    )
    with pytest.raises(UnsupportedCollective, match="process group"):
        CudaCommunicationBackend(extension_status=EXTENSION).compile(
            plan,
            replace(CONTEXT, process_group=object()),
        )


def test_cuda_backend_accepts_async_topology_and_workspace_budget() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    topology = backend.compile(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=True,
        ),
        CONTEXT,
    )
    assert topology.execution_info.async_capable is True
    executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
            workspace_policy=WorkspacePolicy(max_cached_bytes=1024),
        ),
        CONTEXT,
    )

    assert executor.execution_info.fast_path == "cuda_reduced_shard"


def test_cuda_backend_binds_stream_safe_workspace_provider_for_async_shards(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module
    from ccdl_comm.cuda.workspace import CudaWorkspacePool

    captured = {}

    def make_transport(**kwargs):
        captured.update(kwargs)
        return lambda tensor, **operation_kwargs: tensor

    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    pool = CudaWorkspacePool(allocator=lambda key, stream: object())

    def make_pool(**kwargs):
        captured["pool_options"] = kwargs
        return pool

    monkeypatch.setattr(compiler_module, "create_torch_workspace_pool", make_pool)
    executor = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=True,
            workspace_policy=WorkspacePolicy(max_cached_bytes=4096),
        ),
        replace(CONTEXT, workspace_budget_bytes=2048),
    )

    from ccdl_comm.cuda.workspace import CudaShardWorkspaceProvider

    assert isinstance(captured["workspace_cache"], CudaShardWorkspaceProvider)
    assert captured["workspace_cache"].pool_reduced_output is False
    assert captured["chunk_plan"].original_numel == 1024
    assert captured["chunk_plan"].world_size == 2
    assert executor.workspace_pool is captured["workspace_cache"].pool
    assert captured["pool_options"]["max_cached_bytes"] == 2048


def test_reduced_shard_executor_acquires_explicit_pooled_output_and_rejects_disabled_cache(
    monkeypatch,
) -> None:
    import ccdl_comm.cuda.compiler as compiler_module
    from ccdl_comm.cuda.workspace import CudaWorkspacePool

    captured = {}
    allocations = []

    class Buffer:
        def __init__(self, key):
            self.key = key
            self.nbytes = key.estimated_bytes

    class Completion:
        def query(self):
            return True

    class CompletionManager:
        def record_for(self, value, *, stream=None):
            return Completion()

    def make_transport(**kwargs):
        captured.update(kwargs)

        def transport(tensor, **operation_kwargs):
            output = operation_kwargs["out"]
            return ReducedShard(
                shard=output,
                shard_index=0,
                shard_numel=512,
                original_shape=(1024,),
                original_numel=1024,
                world_size=2,
                reduce="mean",
            )

        return transport

    pool = CudaWorkspacePool(
        allocator=lambda key, stream: allocations.append((key, stream)) or Buffer(key),
        max_cached_bytes=4096,
    )
    monkeypatch.setattr(compiler_module, "CudaCompletionManager", lambda **kwargs: CompletionManager())
    monkeypatch.setattr(compiler_module, "create_torch_workspace_pool", lambda **kwargs: pool)
    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
            workspace_policy=WorkspacePolicy(max_cached_bytes=4096),
        ),
        CONTEXT,
    )

    lease = executor.acquire_output()
    reduced = executor.run("bucket", out=lease).wait()

    assert reduced.shard is lease.buffer
    assert allocations[0][0].workspace_kind == "reduced_output"
    assert captured["workspace_cache"].pool_reduced_output is False
    with pytest.raises(RuntimeError, match="after mark_used"):
        lease.release_unused()
    lease.release_after(reduced.shard)

    foreign = executor.acquire_output()
    other_executor = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
            workspace_policy=WorkspacePolicy(max_cached_bytes=4096),
        ),
        CONTEXT,
    )
    with pytest.raises(RuntimeError, match="different executor"):
        other_executor.run("bucket", out=foreign)
    foreign.release_unused()

    disabled = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
            workspace_policy=WorkspacePolicy(cache=False),
        ),
        CONTEXT,
    )
    with pytest.raises(RuntimeError, match="cache is disabled"):
        disabled.acquire_output()
    disabled_lease = executor.acquire_output()
    with pytest.raises(RuntimeError, match="cache is disabled"):
        disabled.run("bucket", out=disabled_lease)
    disabled_lease.release_unused()

    too_small = backend.compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=False,
        ),
        replace(CONTEXT, workspace_budget_bytes=512),
    )
    with pytest.raises(RuntimeError, match="budget cannot represent"):
        too_small.acquire_output()


def test_reduced_shard_executor_retains_leased_output_until_async_work_completes(
    monkeypatch,
) -> None:
    import ccdl_comm.cuda.compiler as compiler_module
    from ccdl_comm.cuda.workspace import CudaWorkspacePool

    class Buffer:
        def __init__(self, key):
            self.nbytes = key.estimated_bytes

    class CompletionManager:
        def record_for(self, value, *, stream=None):
            return type("Completion", (), {"query": lambda self: True})()

    class PendingWork:
        def __init__(self, result):
            self.result = result
            self.ready = False
            self.future = type("Future", (), {})()

        def wait(self):
            self.ready = True
            return self.result

        def query(self):
            return self.ready

        def get_future(self):
            return self.future

    def make_transport(**kwargs):
        def transport(tensor, **operation_kwargs):
            output = operation_kwargs["out"]
            return PendingWork(
                ReducedShard(
                    shard=output,
                    shard_index=0,
                    shard_numel=512,
                    original_shape=(1024,),
                    original_numel=1024,
                    world_size=2,
                    reduce="mean",
                )
            )

        return transport

    monkeypatch.setattr(compiler_module, "CudaCompletionManager", lambda **kwargs: CompletionManager())
    monkeypatch.setattr(
        compiler_module,
        "create_torch_workspace_pool",
        lambda **kwargs: CudaWorkspacePool(allocator=lambda key, stream: Buffer(key)),
    )
    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    executor = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=True,
        ),
        CONTEXT,
    )

    lease = executor.acquire_output()
    work = executor.run("bucket", out=lease)

    assert lease in work.resources
    assert work.query() is False
    with pytest.raises(RuntimeError, match="associated work completes"):
        lease.release_after(lease.buffer)
    with pytest.raises(RuntimeError, match="associated work completes"):
        lease.release_after(type("Completion", (), {"query": lambda self: True})())
    competing = executor.acquire_output()
    assert competing.buffer is not lease.buffer
    competing.release_unused()
    reduced = work.wait()
    assert reduced.shard is lease.buffer
    lease.release_after(reduced.shard)
    future = work.get_future()
    lease_reference = weakref.ref(lease)
    del work
    del lease
    gc.collect()
    assert future is not None
    assert lease_reference() is not None


def test_reduced_shard_executor_rolls_back_lease_after_operation_failure(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module
    from ccdl_comm.cuda.workspace import CudaWorkspacePool

    class Buffer:
        def __init__(self, key):
            self.nbytes = key.estimated_bytes

    class CompletionManager:
        def record_for(self, value, *, stream=None):
            return type("Completion", (), {"query": lambda self: True})()

    def make_transport(**kwargs):
        def transport(tensor, **operation_kwargs):
            raise RuntimeError("transport failed after async cleanup")

        return transport

    pool = CudaWorkspacePool(allocator=lambda key, stream: Buffer(key))
    monkeypatch.setattr(compiler_module, "CudaCompletionManager", lambda **kwargs: CompletionManager())
    monkeypatch.setattr(compiler_module, "create_torch_workspace_pool", lambda **kwargs: pool)
    monkeypatch.setattr(compiler_module, "make_torch_compressed_reduce_scatter_shard", make_transport)
    executor = CudaCommunicationBackend(extension_status=EXTENSION).compile(
        CommunicationPlan(
            "reduce_scatter",
            "compressed",
            compression=CompressionConfig(bit=8),
            output_layout="shard",
            async_op=True,
        ),
        CONTEXT,
    )
    lease = executor.acquire_output()

    with pytest.raises(RuntimeError, match="transport failed"):
        executor.run("bucket", out=lease)

    lease.release_unused()
    replacement = executor.acquire_output()
    assert replacement.buffer is lease.buffer
    replacement.release_unused()
