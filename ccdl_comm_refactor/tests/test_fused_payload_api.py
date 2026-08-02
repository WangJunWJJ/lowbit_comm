import inspect

from ccdl_comm.collectives.all_reduce import compressed_all_reduce
from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.communication.payload_packing import should_fuse_payload
from ccdl_comm.cuda.executors import CudaAllReduceExecutor


class FakeTensor:
    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


def test_compressed_all_reduce_exposes_fused_payload_option() -> None:
    signature = inspect.signature(compressed_all_reduce)

    assert "fuse_payload" in signature.parameters
    assert signature.parameters["fuse_payload"].default is False
    assert "fuse_payload_min_numel" in signature.parameters


def test_ddp_comm_hook_exposes_fused_payload_option() -> None:
    signature = inspect.signature(create_ddp_comm_hook)

    assert "fuse_payload" in signature.parameters
    assert signature.parameters["fuse_payload"].default is False
    assert "fuse_payload_min_numel" in signature.parameters


def test_should_fuse_payload_respects_threshold() -> None:
    assert should_fuse_payload(FakeTensor(4_194_304), enabled=True, min_numel=4_000_000) is True
    assert should_fuse_payload(FakeTensor(1_048_576), enabled=True, min_numel=4_000_000) is False
    assert should_fuse_payload(FakeTensor(4_194_304), enabled=False, min_numel=4_000_000) is False


def test_cuda_executor_exposes_precollected_payload_workspace_api() -> None:
    signature = inspect.signature(CudaAllReduceExecutor.run_precollected_payloads)

    assert tuple(signature.parameters) == (
        "self",
        "payloads",
        "prepared",
        "output",
        "residual",
    )
