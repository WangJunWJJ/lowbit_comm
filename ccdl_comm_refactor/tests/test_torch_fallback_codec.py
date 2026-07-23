import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.quantization.torch_fallback import dequantize_tensor_fallback, quantize_tensor_fallback


def test_torch_fallback_quantizes_and_dequantizes_groupwise_int8_tensor() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float16)
    config = CompressionConfig(bit=8, group_size=64, quant_type="linear")

    payload = quantize_tensor_fallback(tensor, config)
    restored = dequantize_tensor_fallback(payload, tensor.shape, config, dtype="fp16")

    assert isinstance(payload, CompressedPayload)
    assert payload.buffer.dtype == torch.int8
    assert payload.metadata["scales"].shape == (2,)
    assert payload.metadata["original_numel"] == 128
    assert restored.shape == tensor.shape
    assert restored.dtype == torch.float16
    assert float((tensor.float() - restored.float()).norm() / tensor.float().norm()) < 0.02


def test_torch_fallback_dequantize_accepts_collective_positional_dtype() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float16)
    config = CompressionConfig(bit=8, group_size=64, quant_type="linear")

    payload = quantize_tensor_fallback(tensor, config)
    restored = dequantize_tensor_fallback(payload, tensor.shape, config, "fp16")

    assert restored.shape == tensor.shape


def test_torch_fallback_rejects_non_linear_quantization() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="only quant_type='linear'"):
        quantize_tensor_fallback(torch.ones(64), CompressionConfig(quant_type="normal"))
