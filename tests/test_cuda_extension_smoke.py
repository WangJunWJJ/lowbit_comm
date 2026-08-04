import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import (
    allocate_dequantized_buffer,
    dequantize_reduce_tensors,
    dequantize_tensor,
    inplace_dequantize_reduce_mean_update_error_feedback,
    quantize_tensor,
    update_error_feedback_residual,
)


def test_cuda_extension_quantizes_and_dequantizes_fp16_tensor() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    status = load_cuda_extension()
    if not status.available:
        pytest.skip(status.reason or "ccdl_cuda_ops is not available")

    tensor = torch.randn(4096, device="cuda", dtype=torch.float16)
    config = CompressionConfig(bit=8, group_size=64, quant_type="linear")

    quantized = quantize_tensor(tensor, config, extension_status=status)
    restored = dequantize_tensor(quantized, tensor.shape, config, dtype="fp16", extension_status=status)
    torch.cuda.synchronize()

    relative_l2 = (tensor.float() - restored.float()).norm() / tensor.float().norm()
    assert restored.shape == tensor.shape
    assert torch.isfinite(restored).all()
    assert float(relative_l2) < 0.02


def test_cuda_extension_updates_error_feedback_residual_inplace() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    status = load_cuda_extension()
    if not status.available:
        pytest.skip(status.reason or "ccdl_cuda_ops is not available")

    prepared = torch.randn(4096, device="cuda", dtype=torch.float32)
    restored = torch.randn_like(prepared)
    residual = torch.empty_like(prepared)

    result = update_error_feedback_residual(prepared, restored, residual, extension_status=status)
    torch.cuda.synchronize()

    assert result is residual
    torch.testing.assert_close(residual, prepared - restored)


def test_cuda_extension_fuses_dequant_reduce_mean_and_error_feedback_update() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    status = load_cuda_extension()
    if not status.available:
        pytest.skip(status.reason or "ccdl_cuda_ops is not available")

    config = CompressionConfig(bit=8, group_size=64, quant_type="linear")
    rank0 = torch.randn(4096, device="cuda", dtype=torch.float32)
    rank1 = torch.randn(4096, device="cuda", dtype=torch.float32)
    prepared = torch.randn(4096, device="cuda", dtype=torch.float32)
    buffers = [
        quantize_tensor(rank0, config, extension_status=status),
        quantize_tensor(rank1, config, extension_status=status),
    ]
    expected_restored = dequantize_reduce_tensors(
        buffers,
        prepared.shape,
        config,
        dtype="fp32",
        extension_status=status,
        reduce="mean",
    )
    restored = allocate_dequantized_buffer(prepared, prepared.shape, config)
    residual = torch.empty_like(prepared)

    used_fused = inplace_dequantize_reduce_mean_update_error_feedback(
        buffers,
        prepared,
        restored,
        residual,
        config,
        extension_status=status,
        reduce="mean",
    )
    torch.cuda.synchronize()

    assert used_fused is True
    torch.testing.assert_close(restored, expected_restored)
    torch.testing.assert_close(residual, prepared - expected_restored)
