import pytest

from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.torch_transport import (
    TorchDistributedUnavailableError,
    make_torch_async_all_gather,
    make_torch_all_gather,
    make_torch_all_reduce,
    make_torch_tensor_all_reduce,
)


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


def test_torch_tensor_all_reduce_uses_sum_then_divides_for_mean() -> None:
    calls = []

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def __itruediv__(self, value):
            calls.append(("div", value))
            self.value /= value
            return self

    class FakeReduceOp:
        SUM = "SUM"

    class FakeDistributed:
        ReduceOp = FakeReduceOp

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 4

        def all_reduce(self, tensor, op):
            calls.append(("all_reduce", tensor.value, op))
            tensor.value *= 4

    def fake_import(name):
        assert name == "torch.distributed"
        return FakeDistributed()

    tensor = FakeTensor(2.0)
    result = make_torch_tensor_all_reduce(import_module=fake_import)(tensor, "mean")

    assert result is tensor
    assert tensor.value == 2.0
    assert calls == [("all_reduce", 2.0, "SUM"), ("div", 4)]


def test_torch_all_gather_collects_payload_sized_buffers() -> None:
    calls = []

    class FakeBuffer:
        def __init__(self, value):
            self.value = value
            self.shape = (4,)

        def new_empty(self, shape):
            calls.append(("new_empty", shape))
            return FakeBuffer(None)

        def __eq__(self, other):
            return isinstance(other, FakeBuffer) and self.value == other.value

    class FakeDistributed:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather(self, output_list, tensor):
            calls.append(("all_gather", len(output_list), tensor))
            output_list[0].value = "rank0"
            output_list[1].value = "rank1"

    def fake_import(name):
        assert name == "torch.distributed"
        return FakeDistributed()

    transport = make_torch_all_gather(import_module=fake_import)

    gathered = transport(FakeBuffer("local"))

    assert gathered.world_size == 2
    assert gathered.payloads == [FakeBuffer("rank0"), FakeBuffer("rank1")]
    assert calls == [
        ("new_empty", (4,)),
        ("new_empty", (4,)),
        ("all_gather", 2, FakeBuffer("local")),
    ]


def test_async_all_gather_transport_returns_work_with_future() -> None:
    calls = []

    class FakeFuture:
        def __init__(self):
            self.callbacks = []

        def then(self, callback):
            self.callbacks.append(callback)
            return self

    class FakeHandle:
        def __init__(self):
            self.future = FakeFuture()

        def wait(self):
            calls.append("wait")

        def get_future(self):
            calls.append("get_future")
            return self.future

    class FakeDist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather(output_list, buffer, async_op=False):
            calls.append(("all_gather", len(output_list), async_op))
            output_list[0] = "rank0"
            output_list[1] = "rank1"
            return FakeHandle()

    class FakeBuffer:
        shape = (3,)

        def new_empty(self, shape):
            return ("empty", shape)

    def import_module(name):
        assert name == "torch.distributed"
        return FakeDist

    transport = make_torch_async_all_gather(import_module=import_module)
    work = transport(FakeBuffer())

    assert work.get_future() is work.handle.future
    assert work.wait().payloads == ["rank0", "rank1"]
    assert calls == [("all_gather", 2, True), "get_future", "wait"]
