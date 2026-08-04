from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16"):
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def __add__(self, other):
        return FakeTensor([left + right for left, right in zip(self.values, other.values)], dtype=self.dtype)

    def __truediv__(self, value):
        return FakeTensor([item / value for item in self.values], dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values and self.dtype == other.dtype


class FakeAsyncGatherWork:
    def __init__(self, calls):
        self.payloads = [FakeTensor([2.0]), FakeTensor([4.0])]
        self.world_size = 2
        self._calls = calls

    def wait(self):
        self._calls.append("handle_wait")

    def is_completed(self):
        self._calls.append("handle_query")
        return False


class FakeAsyncReduceWork:
    def __init__(self, calls):
        self.payload = CompressedPayload(FakeTensor([6.0]), (1,), "fp16")
        self._calls = calls

    def wait(self):
        self._calls.append("handle_wait")

    def is_completed(self):
        self._calls.append("handle_query")
        return False


def test_compressed_all_gather_defers_dequantization_until_wait(monkeypatch) -> None:
    import ccdl_comm.collectives.all_gather as all_gather_module

    calls = []
    async_work = FakeAsyncGatherWork(calls)
    monkeypatch.setattr(all_gather_module, "make_torch_async_all_gather", lambda: lambda buffer: calls.append(("launch", buffer)) or async_work)

    work = all_gather_module.compressed_all_gather(
        FakeTensor([1.0]),
        config=CompressionConfig(bit=8),
        async_op=True,
        quantize=lambda tensor, config: calls.append("quantize") or CompressedPayload(tensor, tensor.shape, "fp16"),
        dequantize=lambda payload, shape, config, dtype: calls.append(("dequantize", payload.buffer)) or payload.buffer,
    )

    assert calls == ["quantize", ("launch", FakeTensor([1.0]))]
    assert work.query() is False
    assert calls == ["quantize", ("launch", FakeTensor([1.0])), "handle_query"]
    assert work.wait() == [FakeTensor([2.0]), FakeTensor([4.0])]
    assert calls[-3:] == ["handle_wait", ("dequantize", FakeTensor([2.0])), ("dequantize", FakeTensor([4.0]))]


def test_compressed_all_reduce_defers_gather_reduce_until_wait(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as all_reduce_module

    calls = []
    async_work = FakeAsyncGatherWork(calls)
    monkeypatch.setattr(all_reduce_module, "make_torch_async_all_gather", lambda: lambda buffer: calls.append(("launch", buffer)) or async_work)

    work = all_reduce_module.compressed_all_reduce(
        FakeTensor([1.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        strategy="all_gather",
        async_op=True,
        quantize=lambda tensor, config: calls.append("quantize") or CompressedPayload(tensor, tensor.shape, "fp16"),
        dequantize=lambda payload, shape, config, dtype: calls.append(("dequantize", payload.buffer)) or payload.buffer,
    )

    assert calls == ["quantize", ("launch", FakeTensor([1.0]))]
    assert work.wait() == FakeTensor([3.0])
    assert calls[-3:] == ["handle_wait", ("dequantize", FakeTensor([2.0])), ("dequantize", FakeTensor([4.0]))]


def test_compressed_all_reduce_transport_defers_dequantization_until_wait(monkeypatch) -> None:
    import ccdl_comm.collectives.all_reduce as all_reduce_module

    calls = []
    async_work = FakeAsyncReduceWork(calls)
    monkeypatch.setattr(
        all_reduce_module,
        "make_torch_async_all_reduce",
        lambda: lambda payload, op: calls.append(("launch", payload.buffer, op)) or async_work,
    )

    work = all_reduce_module.compressed_all_reduce(
        FakeTensor([1.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        strategy="all_reduce",
        async_op=True,
        world_size=2,
        quantize=lambda tensor, config: calls.append("quantize") or CompressedPayload(tensor, tensor.shape, "fp16"),
        dequantize=lambda payload, shape, config, dtype: calls.append(("dequantize", payload.buffer)) or payload.buffer,
    )

    assert calls == ["quantize", ("launch", FakeTensor([1.0]), "sum")]
    assert work.wait() == FakeTensor([3.0])
    assert calls[-2:] == ["handle_wait", ("dequantize", FakeTensor([6.0]))]
