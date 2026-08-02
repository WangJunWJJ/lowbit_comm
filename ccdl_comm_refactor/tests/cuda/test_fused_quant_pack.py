from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_quantized_buffer,
    inplace_quantize_pack,
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


@pytest.mark.parametrize("dtype_name", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("bit", (8, 4))
@pytest.mark.parametrize("group_size", (16, 32, 64))
@pytest.mark.parametrize("use_residual", (False, True))
def test_fused_quant_pack_matches_existing_payload_for_uneven_input(
    extension_status,
    dtype_name: str,
    bit: int,
    group_size: int,
    use_residual: bool,
) -> None:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]
    config = CompressionConfig(bit=bit, group_size=group_size, compact=True, allow_experimental=bit == 4)
    tensor = torch.linspace(-3.0, 3.0, group_size + 7, device="cuda", dtype=dtype)
    residual = torch.linspace(-0.2, 0.2, tensor.numel(), device="cuda", dtype=dtype) if use_residual else None
    tensor_before = tensor.clone()
    residual_before = residual.clone() if residual is not None else None
    reference_input = tensor if residual is None else tensor + residual
    reference = quantize_tensor(reference_input, config, extension_status=extension_status)
    output = allocate_quantized_buffer(tensor, config, dtype=dtype_name)
    metadata: dict[str, int | bool] = {}

    used_fused = inplace_quantize_pack(
        tensor,
        output,
        residual,
        config,
        metadata,
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    assert used_fused is True
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(tensor, tensor_before, rtol=0, atol=0)
    if residual is not None:
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    assert metadata == {
        "original_numel": tensor.numel(),
        "padded_numel": group_size * 2,
        "padding_numel": group_size * 2 - tensor.numel(),
        "fused_quant_pack": True,
    }


def test_fused_quant_pack_adds_error_feedback_without_prepared_tensor(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    tensor = torch.randn(131, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(tensor) * 0.01
    residual_before = residual.clone()
    reference = quantize_tensor(tensor + residual, config, extension_status=extension_status)
    output = allocate_quantized_buffer(tensor, config, dtype="fp16")

    assert inplace_quantize_pack(
        tensor,
        output,
        residual,
        config,
        {},
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


def test_fused_quant_pack_supports_unaligned_contiguous_views(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    input_storage = torch.randn(132, device="cuda", dtype=torch.float16)
    residual_storage = torch.randn(132, device="cuda", dtype=torch.float16) * 0.01
    tensor = input_storage[1:132]
    residual = residual_storage[1:132]
    assert tensor.is_contiguous()
    assert residual.is_contiguous()
    assert tensor.data_ptr() % 16 != 0
    assert residual.data_ptr() % 16 != 0
    reference = quantize_tensor(tensor + residual, config, extension_status=extension_status)
    output = allocate_quantized_buffer(tensor, config, dtype="fp16")

    assert inplace_quantize_pack(
        tensor,
        output,
        residual,
        config,
        {},
        extension_status=extension_status,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_fused_quant_pack_reuses_output_without_steady_state_allocation(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    tensor = torch.randn(8_388_608, device="cuda", dtype=torch.float16)
    residual = torch.zeros_like(tensor)
    output = allocate_quantized_buffer(tensor, config, dtype="fp16")
    output_ptr = output.data_ptr()

    for _ in range(10):
        assert inplace_quantize_pack(tensor, output, residual, config, {}, extension_status=extension_status)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(100):
        assert inplace_quantize_pack(tensor, output, residual, config, {}, extension_status=extension_status)
    torch.cuda.synchronize()

    assert output.data_ptr() == output_ptr
    assert torch.cuda.memory_allocated() == allocated_before


def test_fused_quant_pack_accepts_empty_tensor(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    tensor = torch.empty(0, device="cuda", dtype=torch.float16)
    output = allocate_quantized_buffer(tensor, config, dtype="fp16")
    metadata: dict[str, int | bool] = {}

    assert inplace_quantize_pack(tensor, output, None, config, metadata, extension_status=extension_status)
    assert output.numel() == 0
    assert metadata["original_numel"] == 0
    assert metadata["padded_numel"] == 0


def test_fused_quant_pack_explicitly_falls_back_for_noncompact_layout(extension_status) -> None:
    config = CompressionConfig(bit=8, group_size=64, compact=False)
    tensor = torch.randn(64, device="cuda", dtype=torch.float16)
    output = allocate_quantized_buffer(tensor, config, dtype="fp16")
    metadata: dict[str, int | bool] = {}

    assert not inplace_quantize_pack(tensor, output, None, config, metadata, extension_status=extension_status)
    assert metadata["fused_quant_pack"] is False
