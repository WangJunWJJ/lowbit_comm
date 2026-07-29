from ccdl_comm.communication.topology_transport import make_native_topology_all_reduce
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    shape = (4,)

    def __init__(self, values=(1.0, 2.0), dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.device = "cuda:0"

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype)

    def __truediv__(self, value):
        return FakeTensor([item / value for item in self.values], dtype=self.dtype)

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


def test_native_topology_transport_selects_p2p_for_four_ranks() -> None:
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
    assert any(call == {"all_gather_into_tensor": True} for call in calls)


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

            class Work:
                def wait(self):
                    return None

            return [Work() for _ in ops]

        def all_gather_into_tensor(self, output, input):
            calls.append({"all_gather_into_tensor": True})

    class Torch:
        @staticmethod
        def empty_like(tensor):
            return FakeTensor([0.0 for _ in tensor.values], dtype=tensor.dtype)

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
