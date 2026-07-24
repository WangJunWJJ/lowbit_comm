import inspect

from ccdl_comm.collectives.all_reduce import compressed_all_reduce
from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook


def test_compressed_all_reduce_exposes_fused_payload_option() -> None:
    signature = inspect.signature(compressed_all_reduce)

    assert "fuse_payload" in signature.parameters
    assert signature.parameters["fuse_payload"].default is False


def test_ddp_comm_hook_exposes_fused_payload_option() -> None:
    signature = inspect.signature(create_ddp_comm_hook)

    assert "fuse_payload" in signature.parameters
    assert signature.parameters["fuse_payload"].default is False
