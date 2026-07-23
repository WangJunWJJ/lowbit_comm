from types import SimpleNamespace

import pytest

from ccdl_comm.ascend.codec import dequantize_tensor_cann, quantize_tensor_cann
from ccdl_comm.ascend.loader import CannExtensionStatus
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import CCDLUnavailableError


class FakeTensor:
    dtype = "torch.float16"
    shape = (4,)


def test_cann_codec_requires_available_extension() -> None:
    status = CannExtensionStatus(False, None, "ccdl_cann_ops is not installed")

    with pytest.raises(CCDLUnavailableError, match="ccdl_cann_ops is not installed"):
        quantize_tensor_cann(object(), CompressionConfig(), extension_status=status)


def test_cann_quantize_wraps_extension_payload() -> None:
    class FakeCann:
        def __init__(self) -> None:
            self.calls = []

        def quantize_linear_int8(self, tensor, group_size):
            self.calls.append((tensor, group_size))
            return SimpleNamespace(buffer="q", scales="s", original_numel=4)

    extension = FakeCann()
    tensor = FakeTensor()

    payload = quantize_tensor_cann(
        tensor,
        CompressionConfig(),
        extension_status=CannExtensionStatus(True, extension),
    )

    assert isinstance(payload, CompressedPayload)
    assert payload.buffer == "q"
    assert payload.shape == (4,)
    assert payload.dtype == "fp16"
    assert payload.metadata == {"scales": "s", "original_numel": 4}
    assert extension.calls == [(tensor, 64)]


def test_cann_dequantize_calls_extension() -> None:
    class FakeCann:
        def __init__(self) -> None:
            self.calls = []

        def dequantize_linear_int8(self, buffer, scales, original_numel, shape, dtype, group_size):
            self.calls.append((buffer, scales, original_numel, shape, dtype, group_size))
            return "restored"

    extension = FakeCann()
    payload = CompressedPayload(buffer="q", shape=(4,), dtype="fp16", metadata={"scales": "s", "original_numel": 4})

    result = dequantize_tensor_cann(
        payload,
        (4,),
        CompressionConfig(),
        "fp16",
        extension_status=CannExtensionStatus(True, extension),
    )

    assert result == "restored"
    assert extension.calls == [("q", "s", 4, (4,), "fp16", 64)]


def test_cann_codec_rejects_unsupported_config() -> None:
    status = CannExtensionStatus(True, object())

    with pytest.raises(ValueError, match="only supports quant_type='linear'"):
        quantize_tensor_cann(object(), CompressionConfig(quant_type="normal"), extension_status=status)
