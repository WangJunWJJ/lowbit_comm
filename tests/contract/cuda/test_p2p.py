from __future__ import annotations

from pathlib import Path

from lowbit_comm.backends.cuda.loader import CudaExtensionStatus
from lowbit_comm.backends.cuda.p2p import (
    CudaQuantizedReceiver,
    CudaQuantizedSender,
    P2PTags,
)
from lowbit_comm.core import DataType, QuantizedWire


def test_logical_p2p_tag_reserves_distinct_metadata_and_payload_tags() -> None:
    first = P2PTags.from_logical(7)
    second = P2PTags.from_logical(8)

    assert first.metadata == 14
    assert first.payload == 15
    assert second.metadata == 16
    assert len({first.metadata, first.payload, second.metadata, second.payload}) == 4


class _Resource:
    def __init__(self, *, shape: tuple[int, ...] = (), numel: int = 0) -> None:
        self.shape = shape
        self.device = "cuda:0"
        self._numel = numel
        self.released = False

    def numel(self) -> int:
        return self._numel

    def __getitem__(self, key: object) -> "_Resource":
        del key
        return self

    def release(self) -> None:
        self.released = True


class _Handle:
    def __init__(self) -> None:
        self.waited = False

    def is_completed(self) -> bool:
        return self.waited

    def wait(self) -> None:
        self.waited = True


class _Dist:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, int]] = []
        self.handles: list[_Handle] = []

    def isend(self, value: object, peer: int, *, group: object, tag: int) -> _Handle:
        self.calls.append((value, peer, tag))
        handle = _Handle()
        self.handles.append(handle)
        return handle


class _Torch:
    uint8 = "uint8"
    int64 = "int64"

    def __init__(self) -> None:
        self.created: list[_Resource] = []

    def empty(self, size: int, *, dtype: object, device: object) -> _Resource:
        del dtype, device
        result = _Resource(numel=size)
        self.created.append(result)
        return result

    def zeros(self, size: int, *, dtype: object, device: object) -> _Resource:
        return self.empty(size, dtype=dtype, device=device)

    def tensor(self, values: object, *, dtype: object, device: object) -> _Resource:
        del dtype, device
        result = _Resource(numel=len(tuple(values)))
        self.created.append(result)
        return result


def test_quantized_isend_owns_metadata_and_payload_until_both_handles_finish() -> None:
    torch = _Torch()
    dist = _Dist()
    source = _Resource(shape=(64,), numel=64)
    sender = CudaQuantizedSender(
        peer=1,
        tag=3,
        dtype=DataType.FP16,
        wire=QuantizedWire(8, 64),
        max_numel=64,
        extension_status=CudaExtensionStatus(True, object()),
        torch=torch,
        dist=dist,
        quantize=lambda source, output, wire, **kwargs: output,
    )

    work = sender.isend(source, layout_generation=4)

    assert work.query() is False
    assert [call[2] for call in dist.calls] == [6, 7]
    assert all(not resource.released for resource in torch.created)
    assert work.wait() is None
    assert all(handle.waited for handle in dist.handles)
    assert all(resource.released for resource in torch.created)


def test_receiver_is_typed_and_does_not_expose_an_object_metadata_path() -> None:
    assert not hasattr(CudaQuantizedReceiver, "recv_object")
    assert not hasattr(CudaQuantizedReceiver, "irecv_object")


def test_p2p_does_not_create_per_operation_completion_threads() -> None:
    source = (
        Path(__file__).parents[3]
        / "src"
        / "lowbit_comm"
        / "backends"
        / "cuda"
        / "p2p.py"
    ).read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in source
    assert "_P2P_COMPLETION_POOL" not in source
    assert "sleep(0)" not in source
    assert ".tolist()" not in source
    assert "decode_dynamic_metadata_into(" in source
    assert "payload[: packet.payload_numel]" in source


def test_quantized_p2p_requires_a_positive_dynamic_shape_bound() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_numel"):
        CudaQuantizedSender(
            peer=1,
            tag=0,
            dtype=DataType.FP16,
            wire=QuantizedWire(8, 64),
            max_numel=0,
            extension_status=CudaExtensionStatus(True, object()),
            torch=_Torch(),
            dist=_Dist(),
        )
