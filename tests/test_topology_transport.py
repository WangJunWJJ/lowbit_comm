from ccdl_comm.communication.topology_transport import make_native_topology_all_reduce, make_native_topology_reduce_scatter_shard
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    shape = (4,)

    def __init__(self, values=(1.0, 2.0), dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.device = "cuda:0"

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype)

    def copy_(self, other, non_blocking=False):
        self.values = other.values
        return self

    def __truediv__(self, value):
        if isinstance(value, FakeTensor):
            divisor = value.values[0] if value.values else 1
            return FakeTensor([item / divisor for item in self.values], dtype=self.dtype)
        return FakeTensor([item / value for item in self.values], dtype=self.dtype)

    def __mul__(self, value):
        if isinstance(value, FakeTensor):
            factor = value.values[0] if value.values else 1
            return FakeTensor([item * factor for item in self.values], dtype=self.dtype)
        return FakeTensor([item * value for item in self.values], dtype=self.dtype)

    def abs(self):
        return FakeTensor([abs(item) for item in self.values], dtype=self.dtype)

    def max(self, dim=None, keepdim=False):
        class MaxResult:
            def __init__(self, values):
                self.values = values

        return MaxResult(FakeTensor([max(self.values)], dtype=self.dtype))

    def to(self, dtype):
        return FakeTensor(self.values, dtype=dtype)

    def reshape(self, shape):
        return self

    def numel(self):
        return len(self.values)

    def chunk(self, chunks):
        chunk_size = len(self.values) // chunks
        return [FakeTensor(self.values[index : index + chunk_size], dtype=self.dtype) for index in range(0, len(self.values), chunk_size)]

    def new_empty(self, shape, dtype=None, device=None):
        return FakeTensor([0.0] * int(shape[0]), dtype=dtype or self.dtype)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item], dtype=self.dtype)
        return self.values[item]

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_native_topology_transport_selects_tree_for_two_ranks() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        import_module_fn=_importer(world_size=2, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    result = transport(
        FakeTensor(),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert result == FakeTensor([0.5, 1.0])
    assert "ccdl.comm" not in [call.get("import") for call in calls if isinstance(call, dict)]
    assert {"quantize": (0.5, 1.0)} in calls


def test_native_topology_transport_selects_ring_for_four_ranks_by_default() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert "ccdl.comm" not in [call.get("import") for call in calls if isinstance(call, dict)]
    peer_rounds = [call["peers"] for call in calls if "peers" in call]
    assert peer_rounds == [(1, 3), (1, 3), (1, 3)]
    assert any(call == {"all_gather_into_tensor": True, "async_op": False} for call in calls)


def test_native_topology_transport_can_force_p2p_for_four_ranks() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="p2p",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    peer_rounds = [call["peers"] for call in calls if "peers" in call]
    assert peer_rounds == [(1, 1), (2, 2), (3, 3)]
    assert any(call == {"all_gather_into_tensor": True, "async_op": False} for call in calls)


def test_native_topology_transport_can_force_ring_for_four_ranks() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="ring",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="sum",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    peer_rounds = [call["peers"] for call in calls if "peers" in call]
    assert peer_rounds == [(1, 3), (1, 3), (1, 3)]
    assert any(call == {"all_gather_into_tensor": True, "async_op": False} for call in calls)
    assert "ccdl.comm" not in [call.get("import") for call in calls if isinstance(call, dict)]


def test_native_topology_transport_can_force_overlap_gather() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="overlap-gather",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    work = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="sum",
        async_op=True,
        dtype="fp16",
        extension_status=None,
    )

    assert work.method == "overlap-gather"
    assert work.query() is False
    assert sum(isinstance(call, dict) and "dequantize" in call for call in calls) == 1
    assert work.wait() == FakeTensor([1.0, 2.0, 3.0, 4.0])
    dequantize_count = sum(isinstance(call, dict) and "dequantize" in call for call in calls)
    assert dequantize_count == 4
    work.wait()
    assert sum(isinstance(call, dict) and "dequantize" in call for call in calls) == dequantize_count
    assert any(call == {"all_gather_into_tensor": True, "async_op": True} for call in calls)


def test_native_topology_transport_can_force_overlap_p2p() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="overlap-p2p",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    work = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="sum",
        async_op=True,
        dtype="fp16",
        extension_status=None,
    )

    assert work.method == "overlap-p2p"
    assert any(call == {"all_gather_into_tensor": True, "async_op": True} for call in calls)


def test_native_topology_transport_can_force_overlap_tree() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="overlap-tree",
        import_module_fn=_importer(world_size=2, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    work = transport(
        FakeTensor([1.0, 2.0]),
        config=CompressionConfig(bit=8),
        op="sum",
        async_op=True,
        dtype="fp16",
        extension_status=None,
    )

    assert work.method == "overlap-tree"
    assert [call["peers"] for call in calls if "peers" in call] == [(1, 1)]


def test_native_topology_transport_can_force_overlap_scale() -> None:
    calls = []
    transport = make_native_topology_all_reduce(
        method="overlap-scale",
        import_module_fn=_importer(world_size=2, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    work = transport(
        FakeTensor([1.0] * 16),
        config=CompressionConfig(bit=8, group_size=16),
        op="sum",
        async_op=True,
        dtype="fp16",
        extension_status=None,
    )

    assert work.method == "overlap-scale"
    work.wait()
    assert ("all_reduce", "max", False) in calls
    assert ("all_reduce", "sum", True) in calls
    assert "torch.mul" in calls


def test_native_topology_reduce_scatter_shard_can_force_ring_for_four_ranks() -> None:
    calls = []
    transport = make_native_topology_reduce_scatter_shard(
        method="ring",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert result.shard_index == 0
    assert result.shard_numel == 1
    assert result.original_shape == (4,)
    assert result.original_numel == 4
    assert result.world_size == 4
    assert result.transport == "topology_ring"
    assert result.metadata["compression_bit"] == 8
    assert [call["peers"] for call in calls if "peers" in call] == [(1, 3), (1, 3), (1, 3)]
    assert "ccdl.comm" not in [call.get("import") for call in calls if isinstance(call, dict)]


def test_native_topology_reduce_scatter_shard_can_force_p2p_for_four_ranks() -> None:
    calls = []
    transport = make_native_topology_reduce_scatter_shard(
        method="p2p",
        import_module_fn=_importer(world_size=4, calls=calls),
        quantize=_quantize(calls),
        dequantize=_dequantize(calls),
    )

    result = transport(
        FakeTensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="sum",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert result.transport == "topology_p2p"
    assert [call["peers"] for call in calls if "peers" in call] == [(1, 1), (2, 2), (3, 3)]


def _importer(world_size: int, calls: list[dict]):
    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return world_size

        def get_rank(self):
            return 0

        def get_process_group_ranks(self, group):
            raise KeyError(group)

        class ReduceOp:
            MAX = "max"
            SUM = "sum"

        class P2POp:
            def __init__(self, fn, tensor, peer):
                self.fn = fn
                self.tensor = tensor
                self.peer = peer

        def isend(self, tensor, peer):
            return None

        def irecv(self, tensor, peer):
            return None

        def batch_isend_irecv(self, ops):
            calls.append({"p2p_ops": len(ops)})
            calls.append({"peers": tuple(op.peer for op in ops)})

            class Work:
                def wait(self):
                    return None

            return [Work() for _ in ops]

        def all_gather_into_tensor(self, output, input, async_op=False):
            calls.append({"all_gather_into_tensor": True, "async_op": async_op})

            class Work:
                def wait(self):
                    return None

            return Work()

        def all_reduce(self, tensor, op=None, async_op=False):
            calls.append(("all_reduce", op, async_op))

            class Work:
                def wait(self):
                    return None

            return Work()

    class Torch:
        int8 = "torch.int8"

        @staticmethod
        def empty_like(tensor):
            return FakeTensor([0.0 for _ in tensor.values], dtype=tensor.dtype)

        @staticmethod
        def mul(left, right, out=None):
            calls.append("torch.mul")
            result = left * right
            if out is not None:
                out.copy_(result)
                return out
            return result

    def import_module(name: str):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return Torch()
        if name == "ccdl.comm":
            raise AssertionError("native topology transport must not import legacy ccdl.comm")
        raise AssertionError(name)

    return import_module


def _quantize(calls):
    def quantize(tensor, config, *, extension_status):
        calls.append({"quantize": tensor.values})
        return FakeTensor(tensor.values)

    return quantize


def _dequantize(calls):
    def dequantize(buffer, shape, config, *, dtype, extension_status, output=None, reduce_op="none"):
        calls.append({"dequantize": buffer.values, "reduce_op": reduce_op})
        return output if output is not None else FakeTensor(buffer.values)

    return dequantize
