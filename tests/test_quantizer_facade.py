from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def numel(self):
        return len(self.values)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


def test_quantizer_facade_quantizes_and_dequantizes_with_config() -> None:
    from ccdl_comm.quantization import Quantizer

    calls = []

    def quantize(tensor, config, *, extension_status=None, output=None):
        calls.append(("quantize", tensor.values, config.bit, output))
        return output or FakeTensor([10.0, 20.0], dtype="torch.uint8")

    def dequantize(buffer, shape, config, *, dtype, extension_status=None, output=None, reduce_op="none"):
        calls.append(("dequantize", buffer.values, shape, config.group_size, dtype, output, reduce_op))
        result = output or FakeTensor([1.0, 2.0])
        return result

    quantizer = Quantizer(
        CompressionConfig(bit=8, group_size=64),
        dtype="fp16",
        quantize_fn=quantize,
        dequantize_fn=dequantize,
    )
    q = quantizer.quantize(FakeTensor([1.0, 2.0]))
    restored = quantizer.dequantize(q, (2,))

    assert q == FakeTensor([10.0, 20.0])
    assert restored == FakeTensor([1.0, 2.0])
    assert quantizer.is_quantized() is True
    assert calls == [
        ("quantize", (1.0, 2.0), 8, None),
        ("dequantize", (10.0, 20.0), (2,), 64, "fp16", None, "none"),
    ]


def test_quantizer_facade_round_trips_dict() -> None:
    from ccdl_comm.quantization import Quantizer

    quantizer = Quantizer(CompressionConfig(bit=8, group_size=32, stochastic=True), dtype="fp32")
    restored = Quantizer.from_dict(quantizer.to_dict())

    assert restored.config == quantizer.config
    assert restored.dtype == "fp32"


def test_quantizer_facade_allocates_quantized_workspace() -> None:
    from ccdl_comm.quantization import Quantizer

    calls = []
    workspace = FakeTensor([0.0, 0.0], dtype="torch.uint8")

    def allocate(tensor, config, *, dtype):
        calls.append(("allocate", tensor.shape, config.bit, dtype))
        return workspace

    quantizer = Quantizer(CompressionConfig(bit=8), dtype="fp16", allocate_fn=allocate)

    assert quantizer.allocate_q(FakeTensor([1.0, 2.0])) is workspace
    assert calls == [("allocate", (2,), 8, "fp16")]
