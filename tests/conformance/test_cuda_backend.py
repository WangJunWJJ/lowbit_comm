from __future__ import annotations

from ccdl_comm import (
    BackendKey,
    BackendRegistry,
    CommunicationBackend,
    CommunicationPlan,
    CompileContext,
    CompressionConfig,
    compile as compile_plan,
)
from ccdl_comm.cuda.backend import CudaCommunicationBackend, register_cuda_backends
from ccdl_comm.cuda.loader import CudaExtensionStatus


CONTEXT = CompileContext(
    rank=0,
    world_size=2,
    device="cuda:0",
    shape=(4096,),
    dtype="float16",
)
EXTENSION = CudaExtensionStatus(True, object())


def test_cuda_backend_satisfies_protocol_and_reports_contextual_capabilities() -> None:
    backend = CudaCommunicationBackend(extension_status=EXTENSION)

    capabilities = backend.capabilities(CONTEXT)

    assert isinstance(backend, CommunicationBackend)
    assert capabilities.available is True
    assert capabilities.collectives == {
        "all_gather",
        "all_reduce",
        "all_to_all",
        "barrier",
        "broadcast",
        "gather",
        "reduce",
        "reduce_scatter",
        "scatter",
    }
    assert {"all_gather", "topology", "compressed"} <= capabilities.strategies
    assert "hierarchical" not in capabilities.strategies
    assert capabilities.dtypes == {"fp16", "bf16", "fp32"}
    assert capabilities.bits == {4, 8}
    assert capabilities.output_layouts == {"full", "shard"}
    assert capabilities.supports_async is True
    assert "hierarchical" not in capabilities.async_strategies
    assert {"all_gather", "topology", "compressed"} <= capabilities.async_strategies
    assert capabilities.verified_strategies == frozenset()
    assert capabilities.supports_dynamic_shape is False


def test_cuda_backend_separates_implemented_from_a6000_verified_strategies() -> None:
    capabilities = CudaCommunicationBackend(extension_status=EXTENSION).capabilities(
        CompileContext(
            rank=0,
            world_size=2,
            device="cuda:0",
            shape=(16_777_216,),
            dtype="float16",
            device_architecture="NVIDIA RTX A6000",
        )
    )

    assert {"all_gather", "topology", "compressed"} <= capabilities.strategies
    assert capabilities.verified_strategies == {"topology", "compressed"}


def test_hierarchical_is_available_but_not_reported_as_async() -> None:
    capabilities = CudaCommunicationBackend(extension_status=EXTENSION).capabilities(
        CompileContext(
            rank=0,
            world_size=4,
            device="cuda:0",
            shape=(4096,),
            dtype="float16",
            local_rank=0,
            local_world_size=2,
            node_id=0,
            node_count=2,
        )
    )

    assert "hierarchical" in capabilities.strategies
    assert "hierarchical" not in capabilities.async_strategies
    assert capabilities.supports_async is True


def test_cuda_backend_registration_supports_core_compile() -> None:
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=EXTENSION)

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8),
        ),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.backend == "cuda"
    assert compiled.execution_info.executed_strategy == "all_gather"
    assert BackendKey("all_reduce", "all_gather", "cuda", "full") in registry
    assert BackendKey("reduce_scatter", "compressed", "cuda", "shard") in registry


def test_registered_async_topology_stays_on_requested_strategy() -> None:
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=EXTENSION)

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=True,
            fallback=("all_gather",),
        ),
        CONTEXT,
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "topology"
    assert compiled.execution_info.fallback_used is False


def test_registered_topology_subgroup_can_fallback_to_group_aware_all_gather() -> None:
    registry = BackendRegistry()
    register_cuda_backends(registry, extension_status=EXTENSION)

    compiled = compile_plan(
        CommunicationPlan(
            "all_reduce",
            "topology",
            compression=CompressionConfig(bit=8),
            async_op=False,
            fallback=("all_gather",),
        ),
        CompileContext(
            rank=0,
            world_size=2,
            device="cuda:0",
            shape=(4096,),
            dtype="float16",
            process_group=object(),
        ),
        registry=registry,
    )

    assert compiled.execution_info.executed_strategy == "all_gather"
    assert compiled.execution_info.fallback_used is True


def test_cuda_package_import_does_not_import_torch() -> None:
    import subprocess
    import sys

    source = "import sys; import ccdl_comm.cuda; assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", source], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
