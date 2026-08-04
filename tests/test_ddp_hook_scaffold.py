from ccdl_comm.communication.ddp import DDPBucketProcessor
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus


class FakeTensor:
    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)

    def __add__(self, other):
        return FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __sub__(self, other):
        return FakeTensor(a - b for a, b in zip(self.values, other.values))

    def detach(self):
        return FakeTensor(self.values)

    def clone(self):
        return FakeTensor(self.values)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


class FakeBucket:
    def __init__(self, index, tensor):
        self._index = index
        self._tensor = tensor

    def index(self):
        return self._index

    def buffer(self):
        return self._tensor


def test_bucket_processor_calls_quantize_and_dequantize_with_bucket_view() -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor, config.bit))
        return {"payload": tensor}

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload, shape, dtype))
        return payload["payload"]

    processor = DDPBucketProcessor(CompressionConfig(bit=8), quantize=quantize, dequantize=dequantize)

    result = processor.process(FakeBucket(7, FakeTensor([1.0, 2.0])), dtype="fp16")

    assert result == FakeTensor([1.0, 2.0])
    assert calls == [
        ("quantize", FakeTensor([1.0, 2.0]), 8),
        ("dequantize", {"payload": FakeTensor([1.0, 2.0])}, (2,), "fp16"),
    ]


def test_bucket_processor_applies_error_feedback_before_quantization() -> None:
    seen = []

    def quantize(tensor, config):
        seen.append(tensor)
        return tensor

    def dequantize(payload, shape, config, dtype):
        if len(seen) == 1:
            return FakeTensor([3.25])
        return payload

    processor = DDPBucketProcessor(CompressionConfig(bit=8, error_feedback=True), quantize=quantize, dequantize=dequantize)
    processor.process(FakeBucket(0, FakeTensor([4.0])), dtype="fp16")
    processor.process(FakeBucket(0, FakeTensor([10.0])), dtype="fp16")

    assert seen == [FakeTensor([4.0]), FakeTensor([10.75])]


def test_bucket_processor_can_be_created_from_cuda_codec() -> None:
    class FakeExtension:
        QuantType = type("QuantType", (), {"Linear": "linear"})()
        DType = type("DType", (), {"FP16": "fp16"})()
        ReduceOP = type("ReduceOP", (), {"NONE": "none"})()

        def quantize(self, *args):
            return {"args": args}

        def dequantize(self, payload, *args):
            return payload["args"][0]

    status = CudaExtensionStatus(available=True, module=FakeExtension())
    processor = DDPBucketProcessor.from_cuda_codec(CompressionConfig(bit=8), extension_status=status)

    result = processor.process(FakeBucket(0, FakeTensor([1.0])), dtype="fp16")

    assert result == FakeTensor([1.0])


def test_bucket_processor_uses_injected_all_reduce_transport() -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return {"buffer": tensor, "shape": tensor.shape, "dtype": "fp16"}

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload, shape, dtype))
        return payload["buffer"]

    def all_reduce(payload, op):
        calls.append(("all_reduce", payload.buffer, op))
        return payload.with_buffer({"buffer": FakeTensor([3.0]), "shape": payload.shape, "dtype": payload.dtype})

    processor = DDPBucketProcessor(
        CompressionConfig(bit=8, error_feedback=False),
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=all_reduce,
    )

    result = processor.process(FakeBucket(0, FakeTensor([1.0])), dtype="fp16")

    assert result == FakeTensor([3.0])
    assert calls == [
        ("quantize", FakeTensor([1.0])),
        ("all_reduce", {"buffer": FakeTensor([1.0]), "shape": (1,), "dtype": "fp16"}, "sum"),
        ("dequantize", {"buffer": FakeTensor([3.0]), "shape": (1,), "dtype": "fp16"}, (1,), "fp16"),
    ]
