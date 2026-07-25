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

    def clone(self):
        return FakeTensor(self.values, dtype=self.dtype)

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


def test_all_gather_hook_bypasses_compression_for_small_buckets() -> None:
    calls = []

    def quantize(tensor, config):
        raise AssertionError("small buckets should not be quantized")

    def dequantize(payload, shape, config, dtype):
        raise AssertionError("small buckets should not be dequantized")

    def bypass_all_reduce(tensor, op):
        calls.append(("bypass", tensor, op))
        return FakeTensor([3.0, 5.0])

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        min_compress_numel=4,
        quantize=quantize,
        dequantize=dequantize,
        bypass_all_reduce=bypass_all_reduce,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([3.0, 5.0])
    assert calls == [("bypass", FakeTensor([1.0, 2.0]), "mean")]


def test_all_gather_hook_uses_default_dequantize_reduce_fastpath(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(2,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(2,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, shape, kwargs["dtype"], kwargs["reduce"]))
        return FakeTensor([2.0, 4.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([2.0, 4.0])
    assert calls == [
        ("dequantize_reduce", ["rank0", "rank1"], (2,), "fp16", "mean"),
    ]


def test_all_gather_hook_skips_error_feedback_for_small_bucket_policy(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(2,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(2,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", kwargs["reduce"]))
        return FakeTensor([2.0, 4.0])

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key))
            return FakeTensor([10.0, 20.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(
            bit=8,
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4,
        ),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        error_feedback=Feedback(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([2.0, 4.0])
    assert ("compensate", 0) not in calls
    assert ("update", 0) not in calls
    assert ("quantize", FakeTensor([1.0, 2.0])) in calls


def test_all_gather_hook_updates_error_feedback_when_policy_allows(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
            ],
            world_size=2,
        )

    def dequantize_reduce(buffers, shape, config, **kwargs):
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([10.0, 20.0, 30.0, 40.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key, original, transmitted))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(
            bit=8,
            error_feedback=True,
            error_feedback_policy="large_bucket_only",
            error_feedback_min_numel=4,
        ),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        error_feedback=Feedback(),
        future_factory=FakeFuture,
    )

    hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert ("compensate", 0, FakeTensor([1.0, 2.0, 3.0, 4.0])) in calls
    assert (
        "update",
        0,
        FakeTensor([10.0, 20.0, 30.0, 40.0]),
        FakeTensor([2.0, 4.0, 6.0, 8.0]),
    ) in calls


def test_all_gather_hook_can_complete_from_async_gather_future(monkeypatch) -> None:
    calls = []

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def __init__(self):
            self.payloads = [
                CompressedPayload(buffer="rank0", shape=(2,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(2,), dtype="fp16"),
            ]
            self.world_size = 2

        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(payloads=self.payloads, world_size=self.world_size)

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, kwargs["reduce"]))
        return FakeTensor([2.0, 4.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=False),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_all_gather=async_all_gather,
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0])))

    assert future.result == FakeTensor([2.0, 4.0])
    assert calls == [
        ("quantize", FakeTensor([1.0, 2.0])),
        ("async_all_gather", "local-buffer"),
        "get_future",
        "then",
        "wait",
        ("dequantize_reduce", ["rank0", "rank1"], "mean"),
    ]


def test_all_gather_hook_keeps_error_feedback_on_sync_path_when_async_requested(monkeypatch) -> None:
    calls = []

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def all_gather(payload):
        calls.append(("all_gather", payload.buffer))
        return GatheredPayloads(
            payloads=[
                CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
            ],
            world_size=2,
        )

    def async_all_gather(buffer):
        raise AssertionError("error feedback buckets should use the synchronous gather path")

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, kwargs["reduce"]))
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([10.0, 20.0, 30.0, 40.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key, original, transmitted))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        all_gather=all_gather,
        async_gather=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert future.result == FakeTensor([2.0, 4.0, 6.0, 8.0])
    assert calls == [
        ("compensate", 0, FakeTensor([1.0, 2.0, 3.0, 4.0])),
        ("quantize", FakeTensor([10.0, 20.0, 30.0, 40.0])),
        ("all_gather", "local-buffer"),
        ("dequantize_reduce", ["rank0", "rank1"], "mean"),
        (
            "update",
            0,
            FakeTensor([10.0, 20.0, 30.0, 40.0]),
            FakeTensor([2.0, 4.0, 6.0, 8.0]),
        ),
    ]


def test_all_gather_hook_can_run_error_feedback_through_async_pipeline(monkeypatch) -> None:
    calls = []

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            calls.append("completion_wait")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([10.0, 20.0, 30.0, 40.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key, original, transmitted))

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, kwargs["reduce"]))
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert future.result == FakeTensor([2.0, 4.0, 6.0, 8.0])
    assert calls == [
        ("compensate", 0, FakeTensor([1.0, 2.0, 3.0, 4.0])),
        ("quantize", FakeTensor([10.0, 20.0, 30.0, 40.0])),
        ("async_all_gather", "local-buffer"),
        "get_future",
        "then",
        "wait",
        ("dequantize_reduce", ["rank0", "rank1"], "mean"),
        (
            "update",
            0,
            FakeTensor([10.0, 20.0, 30.0, 40.0]),
            FakeTensor([2.0, 4.0, 6.0, 8.0]),
        ),
        ("record", FakeTensor([2.0, 4.0, 6.0, 8.0])),
        "completion_wait",
    ]


def test_all_gather_async_error_feedback_skips_cpu_completion_synchronize(monkeypatch) -> None:
    calls = []

    class FakeTorchFuture:
        def then(self, callback):
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            return FakeTorchFuture()

        def wait(self):
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            calls.append("completion_wait")

        def synchronize(self):
            calls.append("completion_synchronize")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            return FakeTensor([10.0, 20.0, 30.0, 40.0])

        def update(self, key, *, original, transmitted):
            calls.append(("update", key, original, transmitted))

    def quantize(tensor, config):
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        return FakeGatherWork()

    def dequantize_reduce(buffers, shape, config, **kwargs):
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert "completion_wait" in calls
    assert "completion_synchronize" not in calls


def test_all_gather_hook_can_use_native_error_feedback_update_for_existing_residual(monkeypatch) -> None:
    calls = []
    residual = FakeTensor([0.5, 0.5, 0.5, 0.5])

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            calls.append("completion_wait")

        def synchronize(self):
            calls.append("completion_synchronize")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([1.5, 2.5, 3.5, 4.5])

        def update(self, key, *, original, transmitted):
            raise AssertionError("native update should replace Python feedback.update")

        def get(self, key):
            calls.append(("get", key))
            return residual

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(buffers, shape, config, **kwargs):
        calls.append(("dequantize_reduce", buffers, kwargs["reduce"]))
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    def native_error_feedback_update(prepared, restored, existing_residual):
        calls.append(("native_update", prepared, restored, existing_residual))

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        native_error_feedback_update=native_error_feedback_update,
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert future.result == FakeTensor([2.0, 4.0, 6.0, 8.0])
    assert ("native_update", FakeTensor([1.5, 2.5, 3.5, 4.5]), FakeTensor([2.0, 4.0, 6.0, 8.0]), residual) in calls
    assert ("get", 0) in calls


def test_all_gather_hook_can_use_combined_native_dequant_reduce_feedback_update(monkeypatch) -> None:
    calls = []
    residual = FakeTensor([0.5, 0.5, 0.5, 0.5])

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            calls.append("completion_wait")

        def synchronize(self):
            calls.append("completion_synchronize")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([1.5, 2.5, 3.5, 4.5])

        def update(self, key, *, original, transmitted):
            raise AssertionError("combined native path should replace Python feedback.update")

        def get(self, key):
            calls.append(("get", key))
            return residual

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(*args, **kwargs):
        raise AssertionError("combined native path should replace separate dequantize_reduce")

    def combined(buffers, prepared, existing_residual, shape, config, **kwargs):
        calls.append(("combined", buffers, prepared, existing_residual, shape, kwargs["dtype"], kwargs["reduce"]))
        return FakeTensor([2.0, 4.0, 6.0, 8.0])

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        native_dequantize_reduce_update_feedback=combined,
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert future.result == FakeTensor([2.0, 4.0, 6.0, 8.0])
    assert (
        "combined",
        ["rank0", "rank1"],
        FakeTensor([1.5, 2.5, 3.5, 4.5]),
        residual,
        (4,),
        "fp16",
        "mean",
    ) in calls
    assert ("get", 0) in calls


def test_all_gather_hook_can_use_inplace_fused_feedback_workspace(monkeypatch) -> None:
    calls = []
    residual = FakeTensor([0.5, 0.5, 0.5, 0.5])
    restored_workspace = FakeTensor([0.0, 0.0, 0.0, 0.0])

    class FakeTorchFuture:
        def then(self, callback):
            calls.append("then")
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            calls.append("get_future")
            return FakeTorchFuture()

        def wait(self):
            calls.append("wait")
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            calls.append("completion_wait")

        def synchronize(self):
            calls.append("completion_synchronize")

    class CompletionManager:
        def record_for(self, tensor):
            calls.append(("record", tensor))
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            calls.append(("compensate", key, tensor))
            return FakeTensor([1.5, 2.5, 3.5, 4.5])

        def update(self, key, *, original, transmitted):
            raise AssertionError("inplace fused path should replace Python feedback.update")

        def get(self, key):
            calls.append(("get", key))
            return residual

    def quantize(tensor, config):
        calls.append(("quantize", tensor))
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        calls.append(("async_all_gather", buffer))
        return FakeGatherWork()

    def dequantize_reduce(*args, **kwargs):
        raise AssertionError("inplace fused path should replace separate dequantize_reduce")

    def allocate_workspace(tensor, shape, config):
        calls.append(("allocate_workspace", tensor, shape, config.group_size))
        return restored_workspace

    def inplace_fused(buffers, prepared, restored, existing_residual, config, **kwargs):
        calls.append(
            (
                "inplace_fused",
                buffers,
                prepared,
                restored,
                existing_residual,
                kwargs["reduce"],
            )
        )
        return True

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        native_inplace_dequantize_reduce_update_feedback=inplace_fused,
        allocate_dequantized_workspace=allocate_workspace,
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    future = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert future.result is restored_workspace
    assert (
        "inplace_fused",
        ["rank0", "rank1"],
        FakeTensor([1.5, 2.5, 3.5, 4.5]),
        restored_workspace,
        residual,
        "mean",
    ) in calls
    assert ("allocate_workspace", FakeTensor([1.5, 2.5, 3.5, 4.5]), (4,), 64) in calls


def test_all_gather_hook_reuses_inplace_fused_feedback_workspace(monkeypatch) -> None:
    calls = []
    residual = FakeTensor([0.5, 0.5, 0.5, 0.5])
    restored_workspace = FakeTensor([0.0, 0.0, 0.0, 0.0])

    class FakeTorchFuture:
        def then(self, callback):
            return callback(self)

    class FakeGatherWork:
        def get_future(self):
            return FakeTorchFuture()

        def wait(self):
            return GatheredPayloads(
                payloads=[
                    CompressedPayload(buffer="rank0", shape=(4,), dtype="fp16"),
                    CompressedPayload(buffer="rank1", shape=(4,), dtype="fp16"),
                ],
                world_size=2,
            )

    class Completion:
        def wait(self):
            pass

        def synchronize(self):
            pass

    class CompletionManager:
        def record_for(self, tensor):
            return Completion()

    class Feedback:
        def compensate(self, key, tensor):
            return FakeTensor([1.5, 2.5, 3.5, 4.5])

        def update(self, key, *, original, transmitted):
            raise AssertionError("inplace fused path should replace Python feedback.update")

        def get(self, key):
            return residual

    def quantize(tensor, config):
        return CompressedPayload(buffer="local-buffer", shape=tensor.shape, dtype="fp16")

    def async_all_gather(buffer):
        return FakeGatherWork()

    def dequantize_reduce(*args, **kwargs):
        raise AssertionError("inplace fused path should replace separate dequantize_reduce")

    def allocate_workspace(tensor, shape, config):
        calls.append(("allocate_workspace", shape, config.group_size))
        return restored_workspace

    def inplace_fused(buffers, prepared, restored, existing_residual, config, **kwargs):
        calls.append(("inplace_fused", restored, existing_residual))
        return True

    monkeypatch.setattr("ccdl_comm.communication.ddp_hook.dequantize_reduce_tensors", dequantize_reduce)

    hook = create_ddp_comm_hook(
        CompressionConfig(bit=8, error_feedback=True, error_feedback_policy="always"),
        dtype="fp16",
        strategy="all_gather",
        reduce="mean",
        quantize=quantize,
        async_gather=True,
        async_error_feedback=True,
        async_all_gather=async_all_gather,
        error_feedback=Feedback(),
        native_inplace_dequantize_reduce_update_feedback=inplace_fused,
        allocate_dequantized_workspace=allocate_workspace,
        completion_manager=CompletionManager(),
        future_factory=FakeFuture,
    )

    first = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))
    second = hook(None, FakeBucket(FakeTensor([1.0, 2.0, 3.0, 4.0])))

    assert first.result is restored_workspace
    assert second.result is restored_workspace
    assert calls.count(("allocate_workspace", (4,), 64)) == 1
    assert calls.count(("inplace_fused", restored_workspace, residual)) == 2
