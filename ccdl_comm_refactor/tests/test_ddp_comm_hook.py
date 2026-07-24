from ccdl_comm.communication.ddp_hook import create_ddp_comm_hook
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.communication.gather_reduce import GatheredPayloads
from ccdl_comm.config import CompressionConfig


class FakeFuture:
    def __init__(self):
        self.result = None

    def set_result(self, result):
        self.result = result


class FakeTensor:
    def __init__(self, values, dtype=None):
        self.values = tuple(values)
        self.shape = (len(self.values),)
        self.dtype = dtype

    def __add__(self, other):
        return FakeTensor(a + b for a, b in zip(self.values, other.values))

    def __truediv__(self, value):
        return FakeTensor(a / value for a in self.values)

    def numel(self):
        return len(self.values)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values


class FakeBucket:
    def __init__(self, tensor):
        self._tensor = tensor

    def index(self):
        return 0

    def buffer(self):
        return self._tensor


def test_create_ddp_comm_hook_returns_future_with_processed_bucket() -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor, config.bit))
        return {"buffer": tensor, "shape": tensor.shape, "dtype": "fp16"}

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload, shape, dtype))
        return payload["buffer"]

    def future_factory():
        return FakeFuture()

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=lambda payload, op: payload,
        future_factory=future_factory,
    )

    future = hook(state=None, bucket=FakeBucket(FakeTensor([1.0, 2.0])))

    assert isinstance(future, FakeFuture)
    assert future.result == FakeTensor([1.0, 2.0])
    assert calls == [
        ("quantize", FakeTensor([1.0, 2.0]), 8),
        ("dequantize", {"buffer": FakeTensor([1.0, 2.0]), "shape": (2,), "dtype": "fp16"}, (2,), "fp16"),
    ]


def test_create_ddp_comm_hook_uses_injected_all_reduce_transport() -> None:
    calls = []

    def quantize(tensor, config):
        return {"buffer": tensor, "shape": tensor.shape, "dtype": "fp16"}

    def dequantize(payload, shape, config, dtype):
        return payload["buffer"]

    def all_reduce(payload, op):
        calls.append((payload.buffer, op))
        return payload.with_buffer({"buffer": FakeTensor([3.0]), "shape": payload.shape, "dtype": payload.dtype})

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=all_reduce,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0])))

    assert future.result == FakeTensor([3.0])
    assert calls == [({"buffer": FakeTensor([1.0]), "shape": (1,), "dtype": "fp16"}, "sum")]


def test_create_ddp_comm_hook_can_use_all_gather_mean_strategy() -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return tensor

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload.buffer, shape, dtype))
        return payload.buffer

    def all_gather(payload):
        calls.append(("all_gather", payload.buffer))
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer=FakeTensor([2.0]), shape=(1,), dtype="fp16"),
                CompressedPayload(buffer=FakeTensor([4.0]), shape=(1,), dtype="fp16"),
            ],
            world_size=2,
        )

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        dequantize=dequantize,
        all_gather=all_gather,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0])))

    assert future.result == FakeTensor([3.0])
    assert calls == [
        ("quantize", FakeTensor([1.0])),
        ("all_gather", FakeTensor([1.0])),
        ("dequantize", FakeTensor([2.0]), (1,), "fp16"),
        ("dequantize", FakeTensor([4.0]), (1,), "fp16"),
    ]


def test_create_ddp_comm_hook_can_infer_bucket_dtype() -> None:
    seen_dtypes = []

    def quantize(tensor, config):
        return tensor

    def dequantize(payload, shape, config, dtype):
        seen_dtypes.append(dtype)
        return payload

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        quantize=quantize,
        dequantize=dequantize,
        all_reduce=lambda payload, op: payload,
        future_factory=FakeFuture,
    )

    hook(None, FakeBucket(FakeTensor([1.0], dtype="torch.float32")))

    assert seen_dtypes == ["fp32"]


def test_create_ddp_comm_hook_applies_ddp_annotations() -> None:
    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        quantize=lambda tensor, config: tensor,
        dequantize=lambda payload, shape, config, dtype: payload,
        all_reduce=lambda payload, op: payload,
        future_factory=FakeFuture,
        annotation_provider=lambda: {
            "state": object,
            "bucket": "GradBucket",
            "return": "FutureTensor",
        },
    )

    assert hook.__annotations__ == {
        "state": object,
        "bucket": "GradBucket",
        "return": "FutureTensor",
    }


def test_all_gather_hook_uses_payload_wrapper_when_fusion_threshold_is_not_met(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        return CompressedPayload(
            buffer=tensor,
            shape=tensor.shape,
            dtype="fp16",
            metadata={"original_numel": tensor.numel()},
        )

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload.buffer, payload.metadata["original_numel"]))
        return payload.buffer

    def tensor_all_gather(buffer):
        calls.append(("all_gather", buffer))
        assert isinstance(buffer, FakeTensor)
        return GatheredPayloads(payloads=[buffer, buffer], world_size=2)

    monkeypatch.setattr(
        "ccdl_comm.communication.ddp_hook.make_torch_all_gather",
        lambda: tensor_all_gather,
    )
    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        quantize=quantize,
        dequantize=dequantize,
        fuse_payload=True,
        fuse_payload_min_numel=4_000_000,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([1.0, 2.0])
    assert calls == [
        ("all_gather", FakeTensor([1.0, 2.0])),
        ("dequantize", FakeTensor([1.0, 2.0]), 2),
        ("dequantize", FakeTensor([1.0, 2.0]), 2),
    ]


def test_all_gather_hook_wraps_raw_codec_buffer_for_default_payload_gather(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return tensor

    def dequantize(payload, shape, config, dtype):
        calls.append(("dequantize", payload.buffer, payload.shape, payload.dtype))
        return payload.buffer

    def tensor_all_gather(buffer):
        calls.append(("all_gather", buffer))
        return GatheredPayloads(payloads=[buffer, buffer], world_size=2)

    monkeypatch.setattr(
        "ccdl_comm.communication.ddp_hook.make_torch_all_gather",
        lambda: tensor_all_gather,
    )
    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        quantize=quantize,
        dequantize=dequantize,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([1.0, 2.0])
    assert calls == [
        ("quantize", FakeTensor([1.0, 2.0])),
        ("all_gather", FakeTensor([1.0, 2.0])),
        ("dequantize", FakeTensor([1.0, 2.0]), (2,), "fp16"),
        ("dequantize", FakeTensor([1.0, 2.0]), (2,), "fp16"),
    ]
