from ccdl_comm.communication.hierarchical_transport import make_torch_hierarchical_all_reduce
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.shape = (1,)
        self.dtype = "float32"

    def __itruediv__(self, divisor):
        self.value /= divisor
        return self

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.value == other.value


class FakeBuffer:
    def __init__(self, label):
        self.label = label
        self.shape = (1,)

    def new_empty(self, shape):
        return FakeBuffer(f"empty{shape}")


class FakeDist:
    class ReduceOp:
        SUM = "SUM"

    def __init__(self, *, rank):
        self.rank = rank
        self.calls = []

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        if group is None:
            return 4
        return len(group)

    def get_rank(self):
        return self.rank

    def new_group(self, ranks):
        group = tuple(ranks)
        self.calls.append(("new_group", group))
        return group

    def all_gather(self, output_list, buffer, group):
        self.calls.append(("all_gather", buffer.label, group))
        for index, rank in enumerate(group):
            output_list[index] = FakeBuffer(f"q{rank}")

    def all_reduce(self, tensor, op, group):
        self.calls.append(("all_reduce", tensor.value, op, group))
        tensor.value = 10.0

    def broadcast(self, tensor, src, group):
        self.calls.append(("broadcast", tensor.value, src, group))
        tensor.value = 10.0


def test_hierarchical_transport_reduces_local_group_then_leaders_then_broadcasts() -> None:
    fake_dist = FakeDist(rank=0)

    def import_module(name):
        assert name == "torch.distributed"
        return fake_dist

    def quantize(tensor, config, *, extension_status=None):
        return FakeBuffer("local")

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status=None, reduce):
        assert [buffer.label for buffer in buffers] == ["q0", "q1"]
        assert reduce == "sum"
        return FakeTensor(3.0)

    transport = make_torch_hierarchical_all_reduce(
        local_group_size=2,
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    result = transport(FakeTensor(1.0), config=CompressionConfig(bit=8), op="mean", async_op=False, dtype="fp32", extension_status=None)

    assert result == FakeTensor(2.5)
    assert ("all_gather", "local", (0, 1)) in fake_dist.calls
    assert ("all_reduce", 3.0, "SUM", (0, 2)) in fake_dist.calls
    assert ("broadcast", 10.0, 0, (0, 1)) in fake_dist.calls


def test_hierarchical_transport_non_leader_skips_inter_group_all_reduce() -> None:
    fake_dist = FakeDist(rank=1)

    def import_module(name):
        assert name == "torch.distributed"
        return fake_dist

    transport = make_torch_hierarchical_all_reduce(
        local_group_size=2,
        import_module=import_module,
        quantize=lambda tensor, config, *, extension_status=None: FakeBuffer("local"),
        dequantize_reduce=lambda buffers, shape, config, **kwargs: FakeTensor(3.0),
    )

    result = transport(FakeTensor(1.0), config=CompressionConfig(bit=8), op="sum", async_op=False, dtype="fp32", extension_status=None)

    assert result == FakeTensor(10.0)
    assert not any(call[0] == "all_reduce" for call in fake_dist.calls)
    assert ("broadcast", 3.0, 0, (0, 1)) in fake_dist.calls
