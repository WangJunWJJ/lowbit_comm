from __future__ import annotations

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import (
    CommunicationProgram,
    CompileContext,
    CompressedAllGather,
    CompressedReduceScatter,
    CompressedReduceScatterAllGather,
    DataType,
    FullTensor,
    HierarchicalCompressed,
    QuantizedWire,
    ReduceMean,
    ReducedShard,
    RuntimeBindings,
)
from lowbit_comm.core.lowered import ExecutorKind


def _cuda_backend() -> CudaBackend:
    return CudaBackend(
        extension_status=CudaExtensionStatus(
            available=True,
            module=object(),
            abi_version=1,
        )
    )


def test_cuda_advertised_algorithms_have_executors() -> None:
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )
    backend = _cuda_backend()
    advertised = backend.capabilities(context).supported_algorithms
    executable = {
        "native",
        "compressed_all_gather",
        "compressed_reduce_scatter",
        "compressed_rs_ag",
    }

    assert advertised <= executable
    assert {kind.value for kind in ExecutorKind} >= {
        "native_all_reduce",
        "compressed_all_gather",
        "reduced_shard",
        "compressed_rs_ag",
    }


def test_cuda_capability_primitive_matches_lowered_execution() -> None:
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )
    backend = _cuda_backend()
    programs = (
        CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            QuantizedWire(8, 64, compact=False),
            CompressedAllGather(),
        ),
        CommunicationProgram(
            ReduceMean(),
            ReducedShard(DataType.FP16, 0),
            QuantizedWire(8, 64, compact=False),
            CompressedReduceScatter(),
        ),
        CommunicationProgram(
            ReduceMean(),
            FullTensor(DataType.FP16),
            QuantizedWire(8, 64, compact=False),
            CompressedReduceScatterAllGather(),
        ),
    )
    capabilities = backend.capabilities(context)

    for program in programs:
        lowered = backend.lower(program, context, RuntimeBindings())
        advertised = {
            spec.physical_primitive
            for spec in capabilities.specifications
            if spec.algorithm == _algorithm_name(program.algorithm)
            and spec.output == _output_name(program.output)
            and spec.bit == 8
            and spec.group_size == 64
            and spec.compact is False
        }
        assert advertised == {lowered.physical_primitive.value}


def test_cuda_only_advertises_fused_quantization_schemas() -> None:
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )

    compressed = tuple(
        specification
        for specification in _cuda_backend().capabilities(context).specifications
        if specification.wire == "quantized"
    )

    assert compressed
    assert {specification.bit for specification in compressed} == {8}
    assert {specification.group_size for specification in compressed} == {64}
    assert all(specification.quant_type == "linear" for specification in compressed)
    assert {
        specification.compact
        for specification in compressed
        if specification.algorithm == "compressed_rs_ag"
    } == {False}


def test_cuda_does_not_advertise_fixed_input_fused_paths_above_eight_ranks() -> None:
    context = CompileContext(
        rank=0,
        world_size=9,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )

    capabilities = _cuda_backend().capabilities(context)

    assert capabilities.supported_algorithms == frozenset({"native"})
    assert capabilities.supported_bits == frozenset()


def test_cuda_advertises_explicit_hierarchical_for_valid_multi_node_topology() -> None:
    context = CompileContext(
        rank=0,
        world_size=64,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
        topology_signature="node_ids="
        + ",".join(str(rank // 8) for rank in range(64)),
        node_count=8,
    )

    hierarchical = tuple(
        spec
        for spec in _cuda_backend().capabilities(context).specifications
        if spec.algorithm == "hierarchical_compressed"
    )

    assert len(hierarchical) == 2
    assert {spec.operation for spec in hierarchical} == {"sum", "mean"}
    assert {spec.output for spec in hierarchical} == {"full_tensor"}
    assert {spec.bit for spec in hierarchical} == {8}
    assert {spec.group_size for spec in hierarchical} == {64}
    assert {spec.compact for spec in hierarchical} == {False}
    assert {spec.physical_primitive for spec in hierarchical} == {
        "hierarchical_compressed_full_tensor"
    }


def test_cuda_hides_hierarchical_for_unknown_or_inconsistent_topology() -> None:
    contexts = (
        CompileContext(
            rank=0,
            world_size=16,
            shape=(4096,),
            dtype=DataType.FP16,
            device_type="cuda",
            node_count=2,
        ),
        CompileContext(
            rank=0,
            world_size=4,
            shape=(4096,),
            dtype=DataType.FP16,
            device_type="cuda",
            topology_signature="node_ids=0,0,0,0",
            node_count=2,
        ),
    )

    for context in contexts:
        assert "hierarchical_compressed" not in (
            _cuda_backend().capabilities(context).supported_algorithms
        )


def _algorithm_name(algorithm: object) -> str:
    return {
        CompressedAllGather: "compressed_all_gather",
        CompressedReduceScatter: "compressed_reduce_scatter",
        CompressedReduceScatterAllGather: "compressed_rs_ag",
        HierarchicalCompressed: "hierarchical_compressed",
    }[type(algorithm)]


def _output_name(output: object) -> str:
    return "reduced_shard" if isinstance(output, ReducedShard) else "full_tensor"
