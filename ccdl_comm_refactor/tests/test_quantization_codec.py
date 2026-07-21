import pytest
from types import SimpleNamespace

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.quantization.codec import CCDLUnavailableError, dequantize_tensor, quantize_tensor


def test_quantize_tensor_raises_clear_error_when_cuda_extension_is_unavailable():
    status = CudaExtensionStatus(available=False, module=None, reason="ccdl_cuda_ops is not installed")

    with pytest.raises(CCDLUnavailableError, match="ccdl_cuda_ops is not installed"):
        quantize_tensor(object(), CompressionConfig(), extension_status=status)


def test_quantize_tensor_calls_extension_with_normalized_config():
    class FakeExtension:
        def __init__(self):
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def quantize(self, *args):
            self.calls.append(args)
            return "quantized"

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    tensor = object()

    result = quantize_tensor(tensor, CompressionConfig(), extension_status=status)

    assert result == "quantized"
    assert extension.calls == [(tensor, 64, 0, False, 8, "linear-enum", False)]


def test_dequantize_tensor_reshapes_extension_output_to_original_shape():
    class Decoded:
        def __init__(self):
            self.shape = None

        def reshape(self, shape):
            self.shape = shape
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.ReduceOP = SimpleNamespace(NONE="none-enum")
            self.calls = []
            self.decoded = Decoded()

        def dequantize(self, *args):
            self.calls.append(args)
            return self.decoded

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    buffer = object()

    result = dequantize_tensor(buffer, (2, 3), CompressionConfig(), dtype="fp16", extension_status=status)

    assert result is extension.decoded
    assert result.shape == (2, 3)
    assert extension.calls == [(buffer, 64, 0, 8, "none-enum", "linear-enum", "fp16-enum", False)]


def test_quantization_codec_rejects_extension_without_required_symbols():
    status = CudaExtensionStatus(available=True, module=object())

    with pytest.raises(CCDLUnavailableError, match="missing required symbol"):
        quantize_tensor(object(), CompressionConfig(), extension_status=status)
