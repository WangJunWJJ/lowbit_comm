from __future__ import annotations

import pytest

from ccdl_comm import CompressionConfig


class FakeTensor:
    def __init__(self, values, *, dtype="torch.float16") -> None:
        self.values = tuple(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.device = "cuda:0"

    def numel(self) -> int:
        return len(self.values)

    def new_empty(self, shape, dtype=None):
        return FakeTensor([0.0] * int(shape[0]), dtype=dtype or self.dtype)

    def copy_(self, other):
        self.values = other.values
        return self


class FakeHandle:
    def __init__(self, *, target=None, payload=None, error=None) -> None:
        self.target = target
        self.payload = payload
        self.error = error
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        if self.error is not None:
            raise self.error
        if self.target is not None and self.payload is not None:
            self.target.copy_(self.payload)

    def is_completed(self) -> bool:
        return False


class FakeDist:
    def __init__(self, *, receive_payload=None, receive_error=None) -> None:
        self.receive_payload = receive_payload
        self.receive_error = receive_error
        self.calls = []

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def isend(self, tensor, dst, group=None, tag=0):
        self.calls.append(("isend", tensor, dst, group, tag))
        return FakeHandle()

    def irecv(self, tensor, src=None, group=None, tag=0):
        self.calls.append(("irecv", tensor, src, group, tag))
        return FakeHandle(
            target=tensor,
            payload=self.receive_payload,
            error=self.receive_error,
        )


def test_compiled_send_freezes_transport_and_retains_payload_metadata() -> None:
    from ccdl_comm.cuda.p2p_executor import compile_p2p_executor

    group = object()
    dist = FakeDist()
    source = FakeTensor([1.0, 2.0])
    executor = compile_p2p_executor(
        direction="send",
        peer=1,
        tensor=source,
        config=CompressionConfig(bit=8),
        group=group,
        tag=17,
        dtype="fp16",
        import_module_fn=_importer(dist),
        quantize=_quantize,
    )

    work = executor.run(source)

    assert executor.peer == 1
    assert executor.tag == 17
    assert executor.group is group
    assert executor.metadata.protocol_version == 1
    assert executor.metadata.shape == (2,)
    assert executor.metadata.dtype == "fp16"
    assert work.resources[0] is source
    assert work.resources[1].values[:2] == (10.0, 20.0)
    assert work.resources[1].numel() == executor.metadata.payload_numel
    assert work.resources[2] is executor.metadata
    assert work.execution_info is executor.execution_info
    assert dist.calls == [("isend", work.resources[1], 1, group, 17)]
    assert work.wait() is None


def test_compiled_receive_propagates_transport_error_from_wait() -> None:
    from ccdl_comm.cuda.p2p_executor import compile_p2p_executor

    expected = RuntimeError("receive failed")
    dist = FakeDist(receive_error=expected)
    output = FakeTensor([0.0, 0.0])
    dequantize_calls = []
    executor = compile_p2p_executor(
        direction="recv",
        peer=0,
        tensor=output,
        config=CompressionConfig(bit=8),
        tag=19,
        dtype="fp16",
        import_module_fn=_importer(dist),
        allocate_quantized=_allocate,
        dequantize=_dequantize(dequantize_calls),
    )

    work = executor.run(output)

    assert work.resources[0] is output
    assert work.resources[2] is executor.metadata
    with pytest.raises(RuntimeError, match="receive failed"):
        work.wait()
    assert dequantize_calls == []
    with pytest.raises(RuntimeError, match="receive failed"):
        work.wait()


def test_existing_p2p_api_accepts_a_precompiled_executor() -> None:
    from ccdl_comm.communication.point_to_point import iqsend, qsend
    from ccdl_comm.cuda.p2p_executor import compile_p2p_executor

    dist = FakeDist()
    source = FakeTensor([3.0, 4.0])
    config = CompressionConfig(bit=8)
    executor = compile_p2p_executor(
        direction="send",
        peer=1,
        tensor=source,
        config=config,
        tag=23,
        dtype="fp16",
        import_module_fn=_importer(dist),
        quantize=_quantize,
    )

    async_work = iqsend(source, 1, config=config, compiled_executor=executor)
    assert async_work.wait() is None
    assert qsend(source, 1, config=config, compiled_executor=executor) is None
    assert [call[4] for call in dist.calls] == [23, 23]


def test_compiled_p2p_factories_are_public() -> None:
    from ccdl_comm import compile_qrecv, compile_qsend

    assert callable(compile_qsend)
    assert callable(compile_qrecv)


def _importer(dist):
    def import_module(name):
        if name == "torch.distributed":
            return dist
        raise AssertionError(name)

    return import_module


def _quantize(tensor, config, *, extension_status=None):
    del config, extension_status
    return FakeTensor(
        [value * 10 for value in tensor.values] + [0.0] * 64,
        dtype="torch.uint8",
    )


def _allocate(tensor, config, *, dtype):
    del config, dtype
    return FakeTensor([0.0] * 66, dtype="torch.uint8")


def _dequantize(calls):
    def dequantize(
        buffer,
        shape,
        config,
        *,
        dtype,
        extension_status=None,
        output=None,
        reduce_op="none",
    ):
        del config, extension_status, reduce_op
        calls.append((buffer.values, shape, dtype))
        output.copy_(FakeTensor([value * 10 for value in buffer.values]))
        return output

    return dequantize
