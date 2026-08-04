from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def numel(self):
        return len(self.values)

    def new_zeros(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype)

    def new_empty(self, shape):
        return FakeTensor([0.0] * int(shape[0]), dtype=self.dtype)

    def reshape(self, shape):
        return self

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item], dtype=self.dtype)
        return self.values[item]

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


class FakeTorch:
    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        values = []
        for tensor in tensors:
            values.extend(tensor.values)
        return FakeTensor(values, dtype=tensors[0].dtype)


def test_compressed_all_gather_dynamic_exchanges_metadata_and_trimmed_payloads() -> None:
    from ccdl_comm.collectives.dynamic_all_gather import compressed_all_gather_dynamic

    calls = []

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, output, obj):
            calls.append(("all_gather_object", obj))
            output[:] = [
                obj,
                {"shape": (3,), "dtype": "fp16", "payload_numel": 3},
            ]

        def all_gather(self, output, input):
            calls.append(("all_gather", input.values))
            output[:] = [input, FakeTensor([20.0, 30.0, 40.0])]

    def import_module(name):
        if name == "torch.distributed":
            return Dist()
        if name == "torch":
            return FakeTorch
        raise AssertionError(name)

    def quantize(tensor, config, *, extension_status=None):
        calls.append(("quantize", tensor.values, config.bit))
        return FakeTensor([10.0, 0.0])

    def dequantize(buffer, shape, config, *, dtype, extension_status=None, output=None, reduce_op="none"):
        calls.append(("dequantize", buffer.values, shape, dtype))
        return FakeTensor(buffer.values[: shape[0]])

    result = compressed_all_gather_dynamic(
        FakeTensor([1.0, 2.0]),
        config=CompressionConfig(bit=8),
        dtype="fp16",
        import_module_fn=import_module,
        quantize=quantize,
        dequantize=dequantize,
    )

    assert result == [FakeTensor([10.0, 0.0]), FakeTensor([20.0, 30.0, 40.0])]
    assert ("all_gather_object", {"shape": (2,), "dtype": "fp16", "payload_numel": 2}) in calls
    assert ("all_gather", (10.0, 0.0, 0.0)) in calls
    assert ("dequantize", (10.0, 0.0), (2,), "fp16") in calls
    assert ("dequantize", (20.0, 30.0, 40.0), (3,), "fp16") in calls


def test_qall_gather_dyn_aliases_dynamic_all_gather() -> None:
    from ccdl_comm import qall_gather_dyn

    assert callable(qall_gather_dyn)
