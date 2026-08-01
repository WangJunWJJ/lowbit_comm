from __future__ import annotations

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
)
from ccdl_comm.cuda.backend import CudaCommunicationBackend
from ccdl_comm.cuda.executors import CudaAllReduceExecutor, CudaReducedShardExecutor
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.exceptions import UnsupportedCollective


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
        },
    )(),
)


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


def test_cuda_backend_compiles_topology_and_gates_legacy_hierarchical_plan() -> None:
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
    assert topology.execution_info.async_capable is False
    with pytest.raises(UnsupportedCollective, match="hierarchical"):
        backend.compile(
            CommunicationPlan(
                "all_reduce",
                "hierarchical",
                compression=CompressionConfig(bit=8),
                stages=(CommunicationStage("local", "all_reduce", "all_gather"),),
                async_op=False,
            ),
            CONTEXT,
        )


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


def test_cuda_backend_rejects_async_topology_and_accepts_workspace_budget() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)
    with pytest.raises(UnsupportedCollective, match="synchronous"):
        backend.compile(
            CommunicationPlan(
                "all_reduce",
                "topology",
                compression=CompressionConfig(bit=8),
                async_op=True,
            ),
            CONTEXT,
        )
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
    assert executor.workspace_pool is captured["workspace_cache"].pool
    assert captured["pool_options"]["max_cached_bytes"] == 2048
