from ccdl_comm.communication.topology_transport import make_legacy_topology_all_reduce
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    shape = (4,)

    def __init__(self, values=(1.0, 2.0), dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_legacy_topology_transport_selects_tree_for_two_ranks() -> None:
    calls = []
    transport = make_legacy_topology_all_reduce(import_module_fn=_importer(world_size=2, calls=calls))

    result = transport(
        FakeTensor(),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert result == FakeTensor()
    assert calls[-1]["method"] == "tree"
    assert calls[-1]["op"] == "mean"


def test_legacy_topology_transport_selects_p2p_for_four_ranks() -> None:
    calls = []
    transport = make_legacy_topology_all_reduce(import_module_fn=_importer(world_size=4, calls=calls))

    transport(
        FakeTensor(),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp16",
        extension_status=None,
    )

    assert calls[-1]["method"] == "p2p"


def _importer(world_size: int, calls: list[dict]):
    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return world_size

    class Comm:
        @staticmethod
        def qall_reduce(tensor, *, op, quantizer, method, keep_self, async_op):
            calls.append(
                {
                    "tensor": tensor,
                    "op": op,
                    "bit": quantizer.bit,
                    "method": method,
                    "keep_self": keep_self,
                    "async_op": async_op,
                }
            )
            return None

    class Quantization:
        class Quantizer:
            def __init__(self, group_size, dim, bit, topk, stochastic, dtype, **kwargs):
                self.group_size = group_size
                self.dim = dim
                self.bit = bit
                self.topk = topk
                self.stochastic = stochastic
                self.dtype = dtype
                self.kwargs = kwargs

    def import_module(name: str):
        if name == "torch.distributed":
            return Dist()
        if name == "ccdl.comm":
            return Comm()
        if name == "ccdl.quantization":
            return Quantization()
        raise AssertionError(name)

    return import_module
