import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import load_cuda_extension
from ccdl_comm.quantization.codec import dequantize_tensor, quantize_tensor


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
