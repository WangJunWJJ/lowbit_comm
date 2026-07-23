import pytest

from ccdl_comm.ascend.codec import dequantize_tensor_cann, quantize_tensor_cann
from ccdl_comm.ascend.loader import load_cann_extension
from ccdl_comm.config import CompressionConfig


def test_cann_extension_quantizes_and_dequantizes_fp16_tensor() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    status = load_cann_extension()
    if not status.available:
        pytest.skip(status.reason or "ccdl_cann_ops is not available")

    torch.npu.set_device(0)
    tensor = torch.randn(4096, device="npu:0", dtype=torch.float16)
    config = CompressionConfig(bit=8, group_size=64, quant_type="linear")

    payload = quantize_tensor_cann(tensor, config, extension_status=status)
    restored = dequantize_tensor_cann(payload, tensor.shape, config, "fp16", extension_status=status)
    torch.npu.synchronize()

    relative_l2 = (tensor.float() - restored.float()).norm() / tensor.float().norm()
    assert payload.buffer.dtype == torch.int8
    assert restored.shape == tensor.shape
    assert torch.isfinite(restored).all()
    assert float(relative_l2) < 0.02
