from __future__ import annotations

import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    dequantize_reduce_tensors,
    inplace_dequantize_reduce_mean_requantize,
    quantize_tensor,
)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def extension_status():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    status = load_cuda_extension()
    if not status.available:
        pytest.fail(status.reason or "CCDL CUDA extension is unavailable")
    return status


@pytest.mark.parametrize("dtype_name", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("num_inputs", (1, 2, 4, 8))
def test_fused_requantize_matches_established_operator_chain(
    extension_status,
    dtype_name: str,
    num_inputs: int,
) -> None:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]
    config = CompressionConfig(bit=8, group_size=64, topk=0, compact=False)
    sources = [
        torch.linspace(-3.0 + rank * 0.125, 3.0 + rank * 0.125, 128, device="cuda", dtype=dtype)
        for rank in range(num_inputs)
    ]
    payloads = [quantize_tensor(source, config, extension_status=extension_status) for source in sources]
    reduced = dequantize_reduce_tensors(
        payloads,
        (128,),
        config,
        dtype=dtype_name,
        extension_status=extension_status,
        reduce="mean",
    )
    reference = quantize_tensor(reduced, config, extension_status=extension_status)
    payload_stride = ((reference.numel() + 15) // 16) * 16
    output = torch.full((payload_stride,), 0xA5, device="cuda", dtype=torch.uint8)

    assert inplace_dequantize_reduce_mean_requantize(
        payloads,
        output,
        config,
        dtype=dtype_name,
        extension_status=extension_status,
        divisor=num_inputs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output[: reference.numel()], reference, rtol=0, atol=0)
    assert torch.count_nonzero(output[reference.numel() :]).item() == 0


@pytest.mark.parametrize(
    "config",
    (
        CompressionConfig(bit=4, allow_experimental=True),
        CompressionConfig(group_size=32),
        CompressionConfig(topk=1),
        CompressionConfig(compact=True),
    ),
)
def test_fused_requantize_declines_non_fast_path_policies(extension_status, config) -> None:
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty(80, device="cuda", dtype=torch.uint8)

    assert not inplace_dequantize_reduce_mean_requantize(
        [payload],
        output,
        config,
        dtype="fp16",
        extension_status=extension_status,
        divisor=1,
    )


def test_fused_requantize_rejects_more_than_eight_inputs(extension_status) -> None:
    config = CompressionConfig()
    source = torch.randn(64, device="cuda", dtype=torch.float16)
    payload = quantize_tensor(source, config, extension_status=extension_status)
    output = torch.empty(80, device="cuda", dtype=torch.uint8)

    assert not inplace_dequantize_reduce_mean_requantize(
        [payload] * 9,
        output,
        config,
        dtype="fp16",
        extension_status=extension_status,
        divisor=9,
    )
