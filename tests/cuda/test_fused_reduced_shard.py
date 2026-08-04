from __future__ import annotations

# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_dequantized_buffer,
    inplace_dequantize_reduce_mean,
    quantize_tensor,
)


@pytest.fixture(scope="module")
def extension_status():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    status = load_cuda_extension()
    if not status.available:
        pytest.fail(status.reason or "CCDL CUDA extension is unavailable")
    return status


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("num_inputs", (1, 2, 3, 4, 5, 8))
@pytest.mark.parametrize("compact", (False, True))
@pytest.mark.parametrize(("reduce", "divisor"), (("sum", 1), ("mean", None)))
def test_fused_reduced_shard_writes_reduction_to_padded_caller_output(
    extension_status, dtype: torch.dtype, num_inputs: int, compact: bool, reduce: str, divisor: int | None
) -> None:
    config = CompressionConfig(bit=8, group_size=64, topk=0, compact=compact)
    sources = [torch.linspace(-2.0 + rank, 2.0 + rank, 65, device="cuda", dtype=dtype) for rank in range(num_inputs)]
    payloads = [quantize_tensor(source, config, extension_status=extension_status) for source in sources]
    output = allocate_dequantized_buffer(sources[0], tuple(sources[0].shape), config)
    output_ptr = output.data_ptr()

    assert inplace_dequantize_reduce_mean(
        payloads, output, config, extension_status=extension_status, reduce=reduce
    )
    torch.cuda.synchronize()

    reference = sum(
        (extension_status.module.dequantize(payload, 64, 0, 8, extension_status.module.ReduceOP.NONE,
                                             extension_status.module.QuantType.Linear,
                                             {torch.float16: extension_status.module.DType.FP16,
                                              torch.bfloat16: extension_status.module.DType.BF16,
                                              torch.float32: extension_status.module.DType.FP32}[dtype], compact)
         for payload in payloads)
    ) / (num_inputs if divisor is None else divisor)
    assert output.data_ptr() == output_ptr
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "config",
    (
        CompressionConfig(bit=4, group_size=64, allow_experimental=True),
        CompressionConfig(group_size=32),
        CompressionConfig(topk=1),
    ),
)
def test_fused_reduced_shard_declines_static_capability_misses(extension_status, config: CompressionConfig) -> None:
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty_like(source)

    assert not inplace_dequantize_reduce_mean([payload], output, config, extension_status=extension_status, reduce="mean")


def test_fused_reduced_shard_declines_non_linear_abi_with_a_valid_linear_payload(extension_status) -> None:
    config = CompressionConfig()
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty_like(source)

    assert not extension_status.module.inplace_dequantize_reduce_mean(
        [payload],
        output,
        64,
        0,
        8,
        extension_status.module.QuantType.Normal,
        False,
        1,
    )


def test_fused_reduced_shard_declines_invalid_outputs_and_rejects_zero_divisor(extension_status) -> None:
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = torch.empty(66, device="cuda", dtype=torch.uint8)
    native = extension_status.module.inplace_dequantize_reduce_mean

    for size in (0, 65, 67):
        invalid_payload = torch.empty(size, device="cuda", dtype=torch.uint8)
        assert not native([invalid_payload], torch.empty_like(source), 64, 0, 8, extension_status.module.QuantType.Linear, False, 1)
    assert not native([payload] * 9, torch.empty_like(source), 64, 0, 8, extension_status.module.QuantType.Linear, False, 1)
    assert not native([payload], torch.empty(64, device="cpu", dtype=torch.float16), 64, 0, 8, extension_status.module.QuantType.Linear, False, 1)
    assert not native([payload], torch.empty(65, device="cuda", dtype=torch.float16), 64, 0, 8, extension_status.module.QuantType.Linear, False, 1)
    assert not native([payload], torch.empty(64, 2, device="cuda", dtype=torch.float16).t(), 64, 0, 8, extension_status.module.QuantType.Linear, False, 1)
    with pytest.raises(RuntimeError, match="divisor must be > 0"):
        native([payload], torch.empty_like(source), 64, 0, 8, extension_status.module.QuantType.Linear, False, 0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_reduced_shard_guards_output_device_before_selecting_current_stream(extension_status) -> None:
    output_device = torch.device("cuda:1")
    payload = torch.empty(66, device=output_device, dtype=torch.uint8)
    output = torch.empty(64, device=output_device, dtype=torch.float16)
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0
        assert extension_status.module.inplace_dequantize_reduce_mean(
            [payload], output, 64, 0, 8, extension_status.module.QuantType.Linear, False, 1
        )
        torch.cuda.synchronize(output_device)
    finally:
        torch.cuda.set_device(previous_device)
