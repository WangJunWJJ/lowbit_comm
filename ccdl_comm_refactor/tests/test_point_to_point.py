from ccdl_comm.config import CompressionConfig


class FakeTensor:
    def __init__(self, values, dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.device = "cuda:0"

    def numel(self):
        return len(self.values)

    def new_empty(self, shape, dtype=None):
        return FakeTensor([0.0] * int(shape[0]), dtype=dtype or self.dtype)

    def copy_(self, other):
        self.values = other.values
        return self

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.values == other.values and self.dtype == other.dtype


def test_qsend_quantizes_then_waits_for_send_completion() -> None:
    from ccdl_comm.communication.point_to_point import qsend

    calls = []
    dist = _dist(calls)
    tensor = FakeTensor([1.0, 2.0])

    qsend(
        tensor,
        dst=1,
        config=CompressionConfig(bit=8),
        tag=7,
        import_module_fn=_importer(dist),
        quantize=_quantize(calls),
    )

    assert calls == [
        ("quantize", (1.0, 2.0), 8),
        ("isend", FakeTensor([10.0, 20.0], dtype="torch.uint8"), 1, None, 7),
        "wait",
    ]


def test_qrecv_waits_for_receive_then_dequantizes_into_output() -> None:
    from ccdl_comm.communication.point_to_point import qrecv

    calls = []
    dist = _dist(calls, recv_payload=FakeTensor([3.0, 4.0], dtype="torch.uint8"))
    output = FakeTensor([0.0, 0.0])

    result = qrecv(
        output,
        src=1,
        config=CompressionConfig(bit=8),
        tag=9,
        dtype="fp16",
        import_module_fn=_importer(dist),
        allocate_quantized=_allocate(calls),
        dequantize=_dequantize(calls),
    )

    assert result is output
    assert output == FakeTensor([30.0, 40.0])
    assert calls == [
        ("allocate", (2,), 8, "fp16"),
        ("irecv", (0.0, 0.0), 1, None, 9),
        "wait",
        ("dequantize", (3.0, 4.0), (2,), "fp16"),
    ]


def test_iqsend_returns_distributed_work() -> None:
    from ccdl_comm.communication.point_to_point import iqsend

    calls = []
    dist = _dist(calls)

    work = iqsend(
        FakeTensor([1.0, 2.0]),
        dst=1,
        config=CompressionConfig(bit=8),
        import_module_fn=_importer(dist),
        quantize=_quantize(calls),
    )

    assert work.resources == (FakeTensor([10.0, 20.0], dtype="torch.uint8"),)
    work.wait()
    assert calls == [
        ("quantize", (1.0, 2.0), 8),
        ("isend", FakeTensor([10.0, 20.0], dtype="torch.uint8"), 1, None, 0),
        "wait",
    ]


def test_iqrecv_wait_receives_then_dequantizes() -> None:
    from ccdl_comm.communication.point_to_point import iqrecv

    calls = []
    dist = _dist(calls, recv_payload=FakeTensor([5.0, 6.0], dtype="torch.uint8"))
    output = FakeTensor([0.0, 0.0])

    work = iqrecv(
        output,
        src=1,
        config=CompressionConfig(bit=8),
        dtype="fp16",
        import_module_fn=_importer(dist),
        allocate_quantized=_allocate(calls),
        dequantize=_dequantize(calls),
    )

    assert output == FakeTensor([0.0, 0.0])
    assert work.wait() is output
    assert output == FakeTensor([50.0, 60.0])
    assert calls == [
        ("allocate", (2,), 8, "fp16"),
        ("irecv", (0.0, 0.0), 1, None, 0),
        "wait",
        ("dequantize", (5.0, 6.0), (2,), "fp16"),
    ]


def test_iqrecv_query_does_not_dequantize_receive_buffer() -> None:
    from ccdl_comm.communication.point_to_point import iqrecv

    calls = []
    dist = _dist(calls, recv_payload=FakeTensor([7.0, 8.0], dtype="torch.uint8"))
    output = FakeTensor([0.0, 0.0])

    work = iqrecv(
        output,
        src=1,
        config=CompressionConfig(bit=8),
        dtype="fp16",
        import_module_fn=_importer(dist),
        allocate_quantized=_allocate(calls),
        dequantize=_dequantize(calls),
    )

    assert work.query() is False
    assert output == FakeTensor([0.0, 0.0])
    assert calls == [
        ("allocate", (2,), 8, "fp16"),
        ("irecv", (0.0, 0.0), 1, None, 0),
        "query",
    ]


def _dist(calls, recv_payload=None):
    class Work:
        def __init__(self, target=None):
            self._target = target

        def wait(self):
            calls.append("wait")
            if self._target is not None and recv_payload is not None:
                self._target.copy_(recv_payload)

        def is_completed(self):
            calls.append("query")
            return False

    class Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def send(self, tensor, dst, group=None, tag=0):
            calls.append(("send", tensor, dst, group, tag))

        def recv(self, tensor, src=None, group=None, tag=0):
            calls.append(("recv", tensor.values, src, group, tag))
            if recv_payload is not None:
                tensor.copy_(recv_payload)

        def isend(self, tensor, dst, group=None, tag=0):
            calls.append(("isend", tensor, dst, group, tag))
            return Work()

        def irecv(self, tensor, src=None, group=None, tag=0):
            calls.append(("irecv", tensor.values, src, group, tag))
            return Work(tensor)

    return Dist()


def _importer(dist):
    def import_module(name):
        if name == "torch.distributed":
            return dist
        raise AssertionError(name)

    return import_module


def _quantize(calls):
    def quantize(tensor, config, *, extension_status=None):
        calls.append(("quantize", tensor.values, config.bit))
        return FakeTensor([value * 10 for value in tensor.values], dtype="torch.uint8")

    return quantize


def _dequantize(calls):
    def dequantize(buffer, shape, config, *, dtype, extension_status=None, output=None, reduce_op="none"):
        calls.append(("dequantize", buffer.values, shape, dtype))
        decoded = FakeTensor([value * 10 for value in buffer.values])
        if output is not None:
            output.copy_(decoded)
            return output
        return decoded

    return dequantize


def _allocate(calls):
    def allocate(tensor, config, *, dtype):
        calls.append(("allocate", tensor.shape, config.bit, dtype))
        return FakeTensor([0.0] * tensor.numel(), dtype="torch.uint8")

    return allocate
