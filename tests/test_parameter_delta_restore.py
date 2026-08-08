from __future__ import annotations

from itertools import count

import pytest

from ccdl_comm.communication.cuda_completion import CudaCompletionManager
from ccdl_comm.communication.parameter_delta import ParameterDeltaShard
from ccdl_comm.communication.parameter_delta_restore import (
    TorchQuantizedParameterDeltaRestore,
)
from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.optim import UpdatedParameterShard


POINTERS = count(3000)
PYTHON_COMPLETION = CudaCompletionManager(
    torch_provider=lambda: None,
    extension_status=CudaExtensionStatus(False, None, "test fallback"),
)


class FakeTensor:
    def __init__(
        self,
        values: int | tuple[float, ...],
        *,
        dtype: str = "fp32",
        device: str = "cuda:0",
    ) -> None:
        self.values = (
            [0.0] * values if isinstance(values, int) else list(values)
        )
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self.shape = (len(self.values),)
        self._pointer = next(POINTERS)

    def numel(self) -> int:
        return len(self.values)

    def data_ptr(self) -> int:
        return self._pointer

    def is_contiguous(self) -> bool:
        return True

    def new_empty(self, shape, *, dtype=None):
        numel = shape[0] if isinstance(shape, tuple) else shape
        return FakeTensor(numel, dtype=dtype or self.dtype, device=self.device)

    def copy_(self, other: "FakeTensor") -> "FakeTensor":
        self.values[:] = other.values
        return self

    def add_(self, other: "FakeTensor") -> "FakeTensor":
        self.values[:] = [
            left + right
            for left, right in zip(self.values, other.values, strict=True)
        ]
        return self


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
        del async_op
        if value.dtype == "uint8":
            self._runtime.calls.append("qwd_all_gather")
        else:
            self._runtime.calls.append("fp_all_gather")
            out.values[:] = self._runtime.gathered_fp
        return FakeHandle(self._runtime.collective_error)


class FakeRuntime:
    def __init__(
        self,
        *,
        supports_qwd: bool = True,
        dequantize_result: bool = True,
        collective_error: BaseException | None = None,
    ) -> None:
        self.world_size = 2
        self.supports_qwd_result = supports_qwd
        self.dequantize_result = dequantize_result
        self.collective_error = collective_error
        self.decoded = (0.25, -0.5, 0.0, 0.0, 0.0, 0.0)
        self.gathered_fp = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
        self.calls: list[str] = []
        self.workspace_pointers: list[tuple[int, int, int]] = []
        self.decode_dtypes: list[tuple[str, str]] = []
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
        self.calls.append("quantize_delta")
        return output

    def quantize_difference(
        self,
        master,
        model,
        config,
        *,
        output,
        valid_numel,
    ):
        del master, model, config, valid_numel
        self.calls.append("quantize_difference")
        return True

    def dequantize_add(self, gathered, out, decoded, config, **kwargs):
        del config
        self.calls.append("dequantize_add")
        self.decode_dtypes.append((decoded.dtype, kwargs.get("dtype", "")))
        self.workspace_pointers.append(
            (gathered.data_ptr(), decoded.data_ptr(), out.data_ptr())
        )
        if not self.dequantize_result:
            return False
        decoded.values[:] = self.decoded
        out.add_(decoded)
        return True

    def supports(self, updated, out, payload_numel):
        del updated, out, payload_numel
        return self.supports_qwd_result

    def overwrite(self, out, gathered):
        self.calls.append("overwrite_refresh")
        return out.copy_(gathered)


def delta_shard() -> ParameterDeltaShard:
    return ParameterDeltaShard(
        shard=FakeTensor((0.25, -0.5, 0.0)),
        shard_index=0,
        shard_numel=3,
        valid_numel=3,
        original_numel=5,
        padded_numel=6,
        world_size=2,
        metadata={"step": 101},
    )


def updated_master() -> UpdatedParameterShard:
    return UpdatedParameterShard(
        shard=FakeTensor((7.0, 8.0, 9.0)),
        shard_index=0,
        shard_numel=3,
        valid_numel=3,
        original_numel=5,
        padded_numel=6,
        world_size=2,
        dtype="fp16",
        metadata={"step": 101},
    )


def restore_for(runtime: FakeRuntime) -> TorchQuantizedParameterDeltaRestore:
    return TorchQuantizedParameterDeltaRestore(
        config=CompressionConfig(compact=True),
        model_dtype="fp16",
        import_module=runtime.import_module,
        quantize=runtime.quantize,
        quantize_difference=runtime.quantize_difference,
        dequantize_add=runtime.dequantize_add,
        overwrite=runtime.overwrite,
        quantized_allocator=runtime.allocate,
        supports_qwd=runtime.supports,
        supports_fused_difference=runtime.supports,
        completion_manager=PYTHON_COMPLETION,
    )


def test_qwd_restore_adds_decoded_delta_to_model_copy() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    model = FakeTensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype="fp16")

    result = restore.restore_delta(delta_shard(), out=model).wait()

    assert result is model
    assert model.values == pytest.approx((1.25, 1.5, 3.0, 4.0, 5.0, 6.0))
    assert runtime.calls == ["quantize_delta", "qwd_all_gather", "dequantize_add"]
    assert restore.last_fast_path == "int8_qwd"


def test_fused_difference_quantizes_master_minus_model_without_delta_workspace(
) -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    master = updated_master()
    model = FakeTensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype="fp16")

    result = restore.restore_difference(
        master,
        model_shard=FakeTensor(tuple(model.values[: master.shard_numel]), dtype="fp16"),
        out=model,
    ).wait()

    assert result is model
    assert runtime.calls == [
        "quantize_difference",
        "qwd_all_gather",
        "dequantize_add",
    ]
    assert restore.last_fast_path == "fused_int8_qwd"


def test_qwd_decodes_fp32_payload_into_fp32_workspace() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)

    restore.restore_delta(
        delta_shard(),
        out=FakeTensor(6, dtype="fp16"),
    ).wait()

    assert runtime.decode_dtypes == [("fp32", "fp32")]


def test_fp_refresh_overwrites_model_copy() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    model = FakeTensor((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype="fp16")

    result = restore.refresh(updated_master(), out=model).wait()

    assert result is model
    assert model.values == pytest.approx(runtime.gathered_fp)
    assert runtime.calls == ["fp_all_gather", "overwrite_refresh"]
    assert restore.last_fast_path == "fp_parameter_refresh"


def test_qwd_capability_rejection_happens_before_collective() -> None:
    runtime = FakeRuntime(supports_qwd=False)
    restore = restore_for(runtime)

    assert restore.supports_qwd(updated_master(), FakeTensor(6, dtype="fp16")) is False
    with pytest.raises(RuntimeError, match="select fp refresh before collective"):
        restore.restore_delta(delta_shard(), out=FakeTensor(6, dtype="fp16"))

    assert runtime.calls == []


def test_qwd_steady_state_reuses_all_workspaces() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    model = FakeTensor(6, dtype="fp16")

    for _ in range(100):
        restore.restore_delta(delta_shard(), out=model).wait()
        restore.refresh(updated_master(), out=model).wait()

    pointers = restore.workspace_pointers()
    assert all(len(values) == 1 for values in pointers.values())
    assert len(set(runtime.workspace_pointers)) == 1


def test_in_flight_qwd_workspace_cannot_be_reused() -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)
    model = FakeTensor(6, dtype="fp16")
    work = restore.restore_delta(delta_shard(), out=model)

    with pytest.raises(RuntimeError, match="qWD workspace is in flight"):
        restore.restore_delta(delta_shard(), out=model)

    work.wait()
    assert restore.restore_delta(delta_shard(), out=model).wait() is model


def test_post_collective_decode_rejection_is_a_hard_error() -> None:
    runtime = FakeRuntime(dequantize_result=False)
    restore = restore_for(runtime)

    with pytest.raises(RuntimeError, match="rejected after INT8 collective"):
        restore.restore_delta(delta_shard(), out=FakeTensor(6, dtype="fp16")).wait()

    assert runtime.calls == ["quantize_delta", "qwd_all_gather", "dequantize_add"]


def test_collective_error_does_not_run_qwd_writeback() -> None:
    runtime = FakeRuntime(collective_error=RuntimeError("collective failed"))
    restore = restore_for(runtime)

    with pytest.raises(RuntimeError, match="collective failed"):
        restore.restore_delta(delta_shard(), out=FakeTensor(6, dtype="fp16")).wait()

    assert runtime.calls == ["quantize_delta", "qwd_all_gather"]


@pytest.mark.parametrize(
    "out",
    (
        FakeTensor(5, dtype="fp16"),
        FakeTensor(6, dtype="fp32"),
        FakeTensor(6, dtype="fp16", device="cpu"),
    ),
)
def test_invalid_qwd_output_is_rejected_before_collective(out: FakeTensor) -> None:
    runtime = FakeRuntime()
    restore = restore_for(runtime)

    with pytest.raises(ValueError):
        restore.restore_delta(delta_shard(), out=out)

    assert runtime.calls == []
