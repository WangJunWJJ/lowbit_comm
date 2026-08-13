from __future__ import annotations

from lowbit_comm.backends.cuda.backend import CudaBackend
from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.core import CompileContext, DataType
from lowbit_comm.core.lowered import ExecutorKind


def test_cuda_advertised_algorithms_have_executors() -> None:
    context = CompileContext(
        rank=0,
        world_size=4,
        shape=(4096,),
        dtype=DataType.FP16,
        device_type="cuda",
    )
    backend = CudaBackend(
        extension_status=CudaExtensionStatus(
            available=True,
            module=object(),
            abi_version=1,
        )
    )
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
