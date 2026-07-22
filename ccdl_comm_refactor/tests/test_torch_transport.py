import pytest

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.torch_transport import TorchDistributedUnavailableError, make_torch_all_reduce


def test_make_torch_all_reduce_raises_clear_error_when_torch_is_missing() -> None:
    def missing_import(name):
        raise ModuleNotFoundError(name)

    transport = make_torch_all_reduce(import_module=missing_import)

    with pytest.raises(TorchDistributedUnavailableError, match="torch.distributed is not available"):
        transport(CompressedPayload(buffer="payload", shape=(1,), dtype="fp16"), "sum")


def test_torch_all_reduce_calls_distributed_all_reduce_on_payload_buffer() -> None:
    calls = []

    class FakeReduceOp:
        SUM = "SUM"

    class FakeDistributed:
        ReduceOp = FakeReduceOp

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def all_reduce(self, tensor, op):
            calls.append((tensor, op))

    def fake_import(name):
        assert name == "torch.distributed"
        return FakeDistributed()

    transport = make_torch_all_reduce(import_module=fake_import)
    payload = CompressedPayload(buffer="buffer", shape=(2,), dtype="fp16")

    result = transport(payload, "sum")

    assert result is payload
    assert calls == [("buffer", "SUM")]
