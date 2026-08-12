from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ccdl_comm import CommunicationPlan, CompileContext, CompressionConfig
from ccdl_comm.cuda.backend import CudaCommunicationBackend
from ccdl_comm.cuda.executors import (
    CudaAllReduceExecutor,
    PrecollectedPayloadExecution,
)
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.execution_info import ExecutionInfo


INFO = ExecutionInfo(
    requested_strategy="all_gather",
    executed_strategy="all_gather",
    backend="cuda",
    fallback_used=False,
    fallback_reason=None,
    stage_names=("quantize", "transport", "dequantize"),
    original_bytes=2048,
    compressed_bytes=1024,
    compression_ratio=2.0,
    workspace_cache_hit=True,
    async_capable=True,
    fast_path="cuda_all_gather",
)


@pytest.fixture(scope="module")
def extension_status():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    status = load_cuda_extension()
    if not status.available:
        pytest.fail(status.reason or "CCDL CUDA extension is unavailable")
    return status


def test_executor_uses_prebound_inplace_fused_operation_and_preserves_output_alias() -> None:
    calls = []
    output = object()

    def run_precollected(payloads, *, prepared, output, residual):
        calls.append((payloads, prepared, output, residual))
        return PrecollectedPayloadExecution(output=output, fused=True)

    executor = CudaAllReduceExecutor(
        lambda tensor: tensor,
        INFO,
        precollected_operation=run_precollected,
    )

    result = executor.run_precollected_payloads(
        ["rank0", "rank1"],
        prepared="prepared",
        output=output,
        residual="residual",
    )

    assert result is output
    assert calls == [(["rank0", "rank1"], "prepared", output, "residual")]
    assert executor.last_execution_info.fast_path == "cuda_fused_dequant_reduce_mean_ef"
    assert executor.last_execution_info.fallback_used is False

    fused_info = executor.last_execution_info
    executor.run_precollected_payloads(
        ["rank0", "rank1"],
        prepared="prepared",
        output=output,
        residual="residual",
    )
    assert executor.last_execution_info is fused_info


def test_executor_records_exact_precollected_fallback_reason() -> None:
    reason = "fused dequant requires group_size=64; received 32"
    fallback_output = object()

    executor = CudaAllReduceExecutor(
        lambda tensor: tensor,
        INFO,
        precollected_operation=lambda *args, **kwargs: PrecollectedPayloadExecution(
            output=fallback_output,
            fused=False,
            fallback_reason=reason,
        ),
    )

    result = executor.run_precollected_payloads(
        ["rank0"],
        prepared="prepared",
        output=fallback_output,
        residual="residual",
    )

    assert result is fallback_output
    assert executor.last_execution_info == replace(
        INFO,
        fallback_used=True,
        fallback_reason=reason,
        fast_path="python_fallback",
    )
    assert executor.last_fallback_record.reason == reason
    assert executor.last_fallback_record.from_path == INFO.fast_path
    assert executor.last_fallback_record.to_path == "python_fallback"
    assert executor.execution_counters.snapshot().fallback_runs == 1


def test_executor_accepts_allocation_free_precollected_status_contract() -> None:
    output = object()
    fused = CudaAllReduceExecutor(
        lambda tensor: tensor,
        INFO,
        precollected_operation=lambda *args, **kwargs: None,
    )
    fallback = CudaAllReduceExecutor(
        lambda tensor: tensor,
        INFO,
        precollected_operation=lambda *args, **kwargs: "runtime constraint",
    )

    assert fused.run_precollected_payloads(
        ["rank0"], prepared="prepared", output=output, residual="residual"
    ) is output
    assert fallback.run_precollected_payloads(
        ["rank0"], prepared="prepared", output=output, residual="residual"
    ) is output
    assert fused.last_execution_info.fallback_used is False
    assert fallback.last_execution_info.fallback_reason == "runtime constraint"


def test_executor_does_not_convert_unexpected_kernel_error_into_fallback() -> None:
    def fail_kernel(*args, **kwargs):
        raise RuntimeError("kernel failed")

    executor = CudaAllReduceExecutor(
        lambda tensor: tensor,
        INFO,
        precollected_operation=fail_kernel,
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        executor.run_precollected_payloads(
            ["rank0"],
            prepared="prepared",
            output="output",
            residual="residual",
        )

    assert executor.execution_counters.snapshot().fallback_runs == 0


def test_executor_rejects_precollected_payloads_when_operation_was_not_bound() -> None:
    executor = CudaAllReduceExecutor(lambda tensor: tensor, INFO)

    try:
        executor.run_precollected_payloads(
            ["rank0"],
            prepared="prepared",
            output="output",
            residual="residual",
        )
    except RuntimeError as error:
        assert "precollected payload operation" in str(error)
    else:
        raise AssertionError("missing precollected operation must not silently use a fake fast path")


def test_compiled_all_gather_executor_calls_inplace_symbol_without_allocating_wrapper() -> None:
    class FakeExtension:
        CompressedWork = object
        NATIVE_WORK_ABI_VERSION = 1
        QuantType = SimpleNamespace(Linear="linear-enum")

        def __init__(self) -> None:
            self.inplace_fused_calls = 0
            self.allocating_wrapper_calls = 0
            self.inplace_fused_args = None

        @staticmethod
        def create_cuda_executor():
            return object()

        def inplace_dequantize_reduce_update_local_error_feedback(self, *args):
            self.inplace_fused_calls += 1
            self.inplace_fused_args = args
            return True

        def dequantize_reduce_update_error_feedback(self, *args):
            self.allocating_wrapper_calls += 1
            raise AssertionError("allocating wrapper must not be used by the executor")

    extension = FakeExtension()
    executor = CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(True, extension),
    ).compile(
        CommunicationPlan(
            "all_reduce",
            "all_gather",
            compression=CompressionConfig(bit=8, group_size=64),
        ),
        CompileContext(
            rank=0,
            world_size=2,
            device="cuda:0",
            shape=(1024,),
            dtype="float16",
        ),
    )
    output = object()

    result = executor.run_precollected_payloads(
        ["rank0", "rank1"],
        prepared="prepared",
        output=output,
        residual="residual",
    )

    assert result is output
    assert extension.inplace_fused_calls == 1
    assert extension.inplace_fused_args[1] == 0
    assert extension.allocating_wrapper_calls == 0


class _FallbackOutput:
    def __init__(self) -> None:
        self.divisors = []

    def div_(self, divisor):
        self.divisors.append(divisor)
        return self


class _FallbackExtension:
    CompressedWork = object
    NATIVE_WORK_ABI_VERSION = 1
    QuantType = SimpleNamespace(
        Linear="linear-enum",
        Normal="normal-enum",
        Uniform="uniform-enum",
        E3M0="e3m0-enum",
        E2M1="e2m1-enum",
    )
    DType = SimpleNamespace(FP16="fp16-enum")
    ReduceOP = SimpleNamespace(NONE="none-enum")

    def __init__(self) -> None:
        self.inplace_fused_calls = 0
        self.fallback_reduce_calls = 0
        self.feedback_calls = 0

    @staticmethod
    def create_cuda_executor():
        return object()

    def inplace_dequantize_reduce_update_local_error_feedback(self, *args):
        self.inplace_fused_calls += 1
        return True

    def dequantize(self, *args):
        return "local-restored"

    def inplace_dequantize_reduce(self, *args):
        self.fallback_reduce_calls += 1

    def inplace_error_feedback_update(self, *args):
        self.feedback_calls += 1


def _compile_executor(
    config: CompressionConfig,
    extension: object,
    *,
    shape: tuple[int, ...] = (1024,),
) -> CudaAllReduceExecutor:
    return CudaCommunicationBackend(
        extension_status=CudaExtensionStatus(True, extension),
    ).compile(
        CommunicationPlan("all_reduce", "all_gather", compression=config),
        CompileContext(
            rank=0,
            world_size=2,
            device="cuda:0",
            shape=shape,
            dtype="float16",
        ),
    )


@pytest.mark.parametrize(
    ("config", "reason"),
    (
        (CompressionConfig(group_size=32), "fused dequant requires group_size=64; received 32"),
        (CompressionConfig(topk=1), "fused dequant requires topk=0; received 1"),
        (
            CompressionConfig(bit=4, allow_experimental=True),
            "fused dequant requires bit=8; received 4",
        ),
        (
            CompressionConfig(quant_type="normal"),
            "fused dequant requires quant_type='linear'; received 'normal'",
        ),
    ),
)
def test_compiled_executor_records_static_fused_constraint_fallback(config, reason) -> None:
    extension = _FallbackExtension()
    executor = _compile_executor(config, extension)
    output = _FallbackOutput()

    assert executor.run_precollected_payloads(
        ["rank0", "rank1"],
        prepared="prepared",
        output=output,
        residual="residual",
    ) is output

    assert extension.inplace_fused_calls == 0
    assert extension.fallback_reduce_calls == 1
    assert extension.feedback_calls == 1
    assert output.divisors == [2]
    assert executor.last_execution_info.fallback_used is True
    assert executor.last_execution_info.fallback_reason == reason


def test_compiled_executor_falls_back_before_native_fused_call_above_eight_payloads() -> None:
    extension = _FallbackExtension()
    executor = _compile_executor(CompressionConfig(), extension)
    output = _FallbackOutput()

    executor.run_precollected_payloads(
        [f"rank{rank}" for rank in range(9)],
        prepared="prepared",
        output=output,
        residual="residual",
    )

    assert extension.inplace_fused_calls == 0
    assert executor.last_execution_info.fallback_reason == (
        "fused dequant supports at most 8 payloads; received 9"
    )


def test_compiled_executor_rejects_payload_with_wrong_byte_count_before_native_call() -> None:
    class Buffer:
        dtype = "torch.uint8"
        device = "cuda:0"

        def numel(self):
            return 1055

    extension = _FallbackExtension()
    executor = _compile_executor(CompressionConfig(), extension)

    with pytest.raises(ValueError, match=r"payload\[0\].*1056 bytes.*received 1055"):
        executor.run_precollected_payloads(
            [Buffer()],
            prepared="prepared",
            output=_FallbackOutput(),
            residual="residual",
        )
    assert extension.inplace_fused_calls == 0
    assert extension.fallback_reduce_calls == 0


def test_fallback_updates_feedback_from_local_reconstruction(monkeypatch) -> None:
    import ccdl_comm.cuda.compiler as compiler_module

    class View:
        def __init__(self):
            self.divisors = []

        def div_(self, divisor):
            self.divisors.append(divisor)
            return self

    class PaddedOutput:
        def div_(self, divisor):
            raise AssertionError("padded output must not be divided or passed to EF update")

    restored_view = View()
    feedback_calls = []
    monkeypatch.setattr(
        compiler_module,
        "dequantize_reduce_tensors",
        lambda *args, **kwargs: restored_view,
    )
    monkeypatch.setattr(
        compiler_module,
        "update_error_feedback_residual",
        lambda prepared, restored, residual, **kwargs: feedback_calls.append(
            (prepared, restored, residual)
        ),
    )
    executor = _compile_executor(CompressionConfig(group_size=32), _FallbackExtension())
    output = PaddedOutput()

    assert executor.run_precollected_payloads(
        ["rank0", "rank1"],
        prepared="prepared",
        output=output,
        residual="residual",
    ) is output
    assert restored_view.divisors == [2]
    assert feedback_calls == [("prepared", "local-restored", "residual")]


def test_cuda_executor_fuses_dequant_reduce_mean_feedback_into_output_workspace(
    extension_status,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor

    config = CompressionConfig(bit=8, group_size=64)
    rank_tensors = [
        torch.linspace(-1.0 + rank * 0.1, 1.0 + rank * 0.1, 4096, device="cuda", dtype=torch.float16)
        for rank in range(4)
    ]
    payloads = [quantize_tensor(tensor, config, extension_status=extension_status) for tensor in rank_tensors]
    decoded = [
        dequantize_tensor(
            payload,
            tuple(rank_tensors[0].shape),
            config,
            dtype="fp16",
            extension_status=extension_status,
        )
        for payload in payloads
    ]
    reference = torch.stack(decoded).float().mean(dim=0).half()
    prepared = rank_tensors[0].clone()
    output = torch.empty_like(prepared)
    residual = torch.empty_like(prepared)
    output_ptr = output.data_ptr()
    residual_ptr = residual.data_ptr()
    executor = _compile_executor(config, extension_status.module, shape=tuple(prepared.shape))

    result = executor.run_precollected_payloads(
        payloads,
        prepared=prepared,
        output=output,
        residual=residual,
    )
    torch.cuda.synchronize()

    assert result is output
    assert output.data_ptr() == output_ptr
    assert residual.data_ptr() == residual_ptr
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual, prepared - decoded[0], rtol=2e-2, atol=2e-2)
    assert executor.last_execution_info.fast_path == "cuda_fused_dequant_reduce_mean_ef"


@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
@pytest.mark.parametrize("reduce", ("sum", "mean"))
@pytest.mark.parametrize("local_input_index", (0, 2))
def test_cuda_fused_feedback_matches_local_reconstruction_for_odd_shape(
    extension_status,
    dtype,
    reduce,
    local_input_index,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from ccdl_comm.quantization.codec import (
        allocate_dequantized_buffer,
        dequantize_tensor,
        inplace_dequantize_reduce_update_local_feedback,
        quantize_tensor,
    )

    torch_dtype = getattr(torch, dtype)
    dtype_name = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[dtype]
    config = CompressionConfig(bit=8, group_size=64)
    rank_tensors = [
        torch.linspace(-1.0, 1.0, 131, device="cuda", dtype=torch_dtype) * (rank + 1)
        for rank in range(3)
    ]
    payloads = [quantize_tensor(tensor, config, extension_status=extension_status) for tensor in rank_tensors]
    decoded = [
        dequantize_tensor(
            payload,
            (131,),
            config,
            dtype=dtype_name,
            extension_status=extension_status,
        )
        for payload in payloads
    ]
    prepared = rank_tensors[local_input_index].clone()
    restored = allocate_dequantized_buffer(prepared, (131,), config)
    residual = torch.empty_like(prepared)

    assert inplace_dequantize_reduce_update_local_feedback(
        payloads,
        local_input_index,
        prepared,
        restored,
        residual,
        config,
        extension_status=extension_status,
        reduce=reduce,
    )
    torch.cuda.synchronize()

    expected_global = torch.stack(decoded).sum(dim=0)
    if reduce == "mean":
        expected_global.div_(len(decoded))
    tolerance = 3e-2 if dtype != "float32" else 1e-5
    torch.testing.assert_close(restored[:131], expected_global, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        residual,
        prepared - decoded[local_input_index],
        rtol=tolerance,
        atol=tolerance,
    )


def test_cuda_native_fused_feedback_guards_tensor_device_and_preserves_results(
    extension_status,
) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor

    previous_device = torch.cuda.current_device()
    target = torch.device("cuda:1")
    config = CompressionConfig(bit=8, group_size=64)
    try:
        torch.cuda.set_device(target)
        rank_tensors = [
            torch.linspace(-1.0 + rank * 0.1, 1.0 + rank * 0.1, 4096, device=target, dtype=torch.float16)
            for rank in range(2)
        ]
        payloads = [quantize_tensor(tensor, config, extension_status=extension_status) for tensor in rank_tensors]
        decoded = [
            dequantize_tensor(
                payload,
                tuple(rank_tensors[0].shape),
                config,
                dtype="fp16",
                extension_status=extension_status,
            )
            for payload in payloads
        ]
        reference = torch.stack(decoded).float().mean(dim=0).half()
        prepared = rank_tensors[0].clone()
        output = torch.empty_like(prepared)
        residual = torch.empty_like(prepared)

        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0
        assert extension_status.module.inplace_dequantize_reduce_update_local_error_feedback(
            payloads,
            0,
            prepared,
            output,
            residual,
            64,
            0,
            8,
            extension_status.module.QuantType.Linear,
            False,
            len(payloads),
        )
        assert torch.cuda.current_device() == 0
        torch.cuda.synchronize(target)

        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(residual, prepared - decoded[0], rtol=2e-2, atol=2e-2)
    finally:
        torch.cuda.set_device(previous_device)


def test_cuda_native_fused_feedback_rejects_cross_device_workspace(extension_status) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    prepared = torch.empty(64, device="cuda:1", dtype=torch.float16)
    restored = torch.empty(64, device="cuda:0", dtype=torch.float16)
    residual = torch.empty(64, device="cuda:1", dtype=torch.float16)
    payload = torch.empty(66, device="cuda:0", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="prepared and restored must be on the same device"):
        extension_status.module.inplace_dequantize_reduce_update_local_error_feedback(
            [payload],
            0,
            prepared,
            restored,
            residual,
            64,
            0,
            8,
            extension_status.module.QuantType.Linear,
            False,
            1,
        )


def test_cuda_executor_fused_dequant_has_one_main_launch_and_zero_steady_allocation(
    extension_status,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from ccdl_comm.quantization.codec import quantize_tensor

    config = CompressionConfig(bit=8, group_size=64)
    prepared = torch.randn(1 << 20, device="cuda", dtype=torch.float16)
    payloads = [
        quantize_tensor(prepared + rank * 0.01, config, extension_status=extension_status)
        for rank in range(4)
    ]
    output = torch.empty_like(prepared)
    residual = torch.empty_like(prepared)
    executor = _compile_executor(config, extension_status.module, shape=tuple(prepared.shape))
    for _ in range(5):
        executor.run_precollected_payloads(
            payloads,
            prepared=prepared,
            output=output,
            residual=residual,
        )
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    executor.run_precollected_payloads(
        payloads,
        prepared=prepared,
        output=output,
        residual=residual,
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        executor.run_precollected_payloads(
            payloads,
            prepared=prepared,
            output=output,
            residual=residual,
        )
    torch.cuda.synchronize()
    kernel_names = [event.key for event in profile.key_averages()]
    fused_launches = [name for name in kernel_names if "dequant_reduce_mean_feedback_fused" in name]
    per_rank_launches = [
        name
        for name in kernel_names
        if "dequant_reduce_fused_" in name and "mean_feedback" not in name
    ]
    assert len(fused_launches) == 1
    assert per_rank_launches == []


def test_cuda_native_fused_dequant_rejects_short_payload_without_launch(extension_status) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from ccdl_comm.quantization.codec import quantize_tensor

    config = CompressionConfig()
    prepared = torch.randn(131, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(prepared, config, extension_status=extension_status)
    output = torch.empty_like(prepared)
    residual = torch.empty_like(prepared)

    assert not extension_status.module.inplace_dequantize_reduce_update_local_error_feedback(
        [payload[:-1]],
        0,
        prepared,
        output,
        residual,
        64,
        0,
        8,
        extension_status.module.QuantType.Linear,
        False,
        1,
    )


def test_cuda_executor_fallback_handles_non_group_aligned_shape(extension_status) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from ccdl_comm.quantization.codec import (
        allocate_dequantized_buffer,
        dequantize_tensor,
        quantize_tensor,
    )

    config = CompressionConfig(group_size=32)
    rank_tensors = [
        torch.randn(131, device="cuda", dtype=torch.float16) + rank * 0.01
        for rank in range(2)
    ]
    payloads = [quantize_tensor(tensor, config, extension_status=extension_status) for tensor in rank_tensors]
    decoded = [
        dequantize_tensor(
            payload,
            (131,),
            config,
            dtype="fp16",
            extension_status=extension_status,
        )
        for payload in payloads
    ]
    reference = torch.stack(decoded).float().mean(dim=0).half()
    prepared = rank_tensors[0].clone()
    output = allocate_dequantized_buffer(prepared, (131,), config)
    residual = torch.empty_like(prepared)
    executor = CudaCommunicationBackend(extension_status=extension_status).compile(
        CommunicationPlan("all_reduce", "all_gather", compression=config),
        CompileContext(
            rank=0,
            world_size=2,
            device="cuda:0",
            shape=(131,),
            dtype="fp16",
        ),
    )

    assert executor.run_precollected_payloads(
        payloads,
        prepared=prepared,
        output=output,
        residual=residual,
    ) is output
    torch.cuda.synchronize()

    torch.testing.assert_close(output[:131], reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual, prepared - decoded[0], rtol=2e-2, atol=2e-2)
    assert executor.last_execution_info.fallback_reason == (
        "fused dequant requires group_size=64; received 32"
    )
