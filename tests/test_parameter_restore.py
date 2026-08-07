from __future__ import annotations

from itertools import count

import pytest

from ccdl_comm.communication.parameter_restore import TorchCompressedParameterRestore
from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.optim import UpdatedParameterShard


POINTERS = count(1000)
PYTHON_COMPLETION = CudaCompletionManager(
    torch_provider=lambda: None,
    extension_status=CudaExtensionStatus(False, None, "test fallback"),
)


class FakeTensor:
    def __init__(
        self,
        numel: int,
        *,
        dtype: str = "fp32",
        device: str = "cuda:0",
    ) -> None:
        self._numel = numel
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self.shape = (numel,)
        self._pointer = next(POINTERS)

    def numel(self) -> int:
        return self._numel

    def data_ptr(self) -> int:
        return self._pointer

    def is_contiguous(self) -> bool:
        return True

    def new_empty(self, shape, *, dtype=None):
        numel = shape[0] if isinstance(shape, tuple) else shape
        return FakeTensor(numel, dtype=dtype or self.dtype, device=self.device)


class FakeHandle:
    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error
        self.completed = False

    def wait(self) -> None:
        if self._error is not None:
            raise self._error
        self.completed = True

    def is_completed(self) -> bool:
        return self.completed


class FakeDistributed:
    def __init__(self, runtime: "FakeRuntime") -> None:
        self._runtime = runtime

    def get_world_size(self) -> int:
        return self._runtime.world_size

    def all_gather_into_tensor(self, out, value, *, async_op):
        del out, async_op
        label = (
            "all_gather_into_tensor"
            if value.dtype == "uint8"
            else "fp_all_gather_into_tensor"
        )
        self._runtime.calls.append(label)
        return FakeHandle(self._runtime.collective_error)


class FakeRuntime:
    def __init__(
        self,
        *,
        world_size: int = 2,
        dequantize_supported: bool = True,
        dequantize_result: bool = True,
        collective_error: BaseException | None = None,
    ) -> None:
        self.world_size = world_size
        self.dequantize_supported = dequantize_supported
        self.dequantize_result = dequantize_result
        self.collective_error = collective_error
        self.calls: list[str] = []
        self.send_pointers: list[int] = []
        self.gather_pointers: list[int] = []
        self.distributed = FakeDistributed(self)

    def import_module(self, name: str):
        if name == "torch.distributed":
            return self.distributed
        raise AssertionError(f"unexpected import: {name}")

    def allocate(self, shard, config, *, dtype):
        del config, dtype
        return shard.new_empty((16,), dtype="uint8")

    def quantize(self, shard, config, *, output):
        del shard, config
        self.calls.append("quantize")
        self.send_pointers.append(output.data_ptr())
        return output

    def dequantize_gathered(self, gathered, out, config, **kwargs):
        del out, config, kwargs
        self.calls.append("dequantize_gathered")
        self.gather_pointers.append(gathered.data_ptr())
        return self.dequantize_result

    def supports(self, updated, out, payload_numel):
        del updated, out, payload_numel
        return self.dequantize_supported


def updated() -> UpdatedParameterShard:
    return UpdatedParameterShard(
        shard=FakeTensor(3),
        shard_index=0,
        shard_numel=3,
        valid_numel=3,
        original_numel=5,
        padded_numel=6,
        world_size=2,
        dtype="fp32",
    )


def restore_for(runtime: FakeRuntime) -> TorchCompressedParameterRestore:
    return TorchCompressedParameterRestore(
        config=CompressionConfig(compact=True),
        dtype="fp32",
        import_module=runtime.import_module,
        quantize=runtime.quantize,
        dequantize_gathered=runtime.dequantize_gathered,
        quantized_allocator=runtime.allocate,
        supports_compressed=runtime.supports,
        completion_manager=PYTHON_COMPLETION,
    )


def test_restore_quantizes_gathers_and_dequantizes_directly_into_out() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    out = FakeTensor(updated().padded_numel)

    work = restore.restore(updated(), out=out, async_op=False)

    assert work.wait() is out
    assert runtime.calls == [
        "quantize",
        "all_gather_into_tensor",
        "dequantize_gathered",
    ]


def test_capability_rejection_uses_fp_gather_without_partial_int8_writeback() -> None:
    runtime = FakeRuntime(dequantize_supported=False)
    restore = restore_for(runtime)
    out = FakeTensor(updated().padded_numel)

    result = restore.restore(updated(), out=out, async_op=False).wait()

    assert result is out
    assert runtime.calls == ["fp_all_gather_into_tensor"]


def test_steady_state_reuses_send_and_gather_workspaces() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    shard = updated()
    out = FakeTensor(shard.padded_numel)

    for _ in range(100):
        restore.restore(shard, out=out).wait()

    assert len(set(runtime.send_pointers)) == 1
    assert len(set(runtime.gather_pointers)) == 1


def test_in_flight_workspace_cannot_be_reused() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    shard = updated()
    out = FakeTensor(shard.padded_numel)
    work = restore.restore(shard, out=out)

    with pytest.raises(RuntimeError, match="parameter restore workspace is in flight"):
        restore.restore(shard, out=out)

    work.wait()
    assert restore.restore(shard, out=out).wait() is out


def test_kernel_capability_change_after_collective_is_a_hard_error() -> None:
    runtime = FakeRuntime(dequantize_result=False)
    restore = restore_for(runtime)

    with pytest.raises(RuntimeError, match="rejected after INT8 collective"):
        restore.restore(updated(), out=FakeTensor(6)).wait()

    assert runtime.calls == [
        "quantize",
        "all_gather_into_tensor",
        "dequantize_gathered",
    ]


def test_collective_error_propagates_without_running_kernel() -> None:
    error = RuntimeError("collective failed")
    runtime = FakeRuntime(collective_error=error)
    restore = restore_for(runtime)

    with pytest.raises(RuntimeError, match="collective failed"):
        restore.restore(updated(), out=FakeTensor(6)).wait()

    assert runtime.calls == ["quantize", "all_gather_into_tensor"]


@pytest.mark.parametrize(
    "out",
    (
        FakeTensor(5),
        FakeTensor(6, dtype="fp16"),
        FakeTensor(6, device="cpu"),
    ),
)
def test_invalid_output_is_rejected_before_collective(out: FakeTensor) -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)

    with pytest.raises(ValueError):
        restore.restore(updated(), out=out)

    assert runtime.calls == []
