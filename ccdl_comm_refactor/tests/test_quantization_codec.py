import pytest
from types import SimpleNamespace

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.quantization.codec import (
    CCDLUnavailableError,
    _pad_tensor_to_group_size,
    allocate_dequantized_buffer,
    allocate_quantized_buffer,
    dequantize_reduce_tensors,
    dequantize_tensor,
    quantize_tensor,
    update_error_feedback_residual,
)


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


def test_quantize_tensor_can_enable_compact_kernel_layout():
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

    quantize_tensor(tensor, CompressionConfig(compact=True), extension_status=status)

    assert extension.calls == [(tensor, 64, 0, False, 8, "linear-enum", True)]


def test_quantize_tensor_uses_inplace_extension_when_output_is_provided():
    class FakeExtension:
        def __init__(self):
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_quantize(self, *args):
            self.calls.append(args)

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    tensor = object()
    output = object()

    result = quantize_tensor(tensor, CompressionConfig(compact=True), extension_status=status, output=output)

    assert result is output
    assert extension.calls == [(tensor, output, 64, 0, False, 8, "linear-enum", True)]


def test_pad_tensor_to_group_size_extends_flat_tensor_to_group_boundary():
    class FakeFlatTensor:
        def __init__(self, values):
            self.values = tuple(values)
            self.shape = (len(self.values),)

        def new_zeros(self, shape):
            return FakeFlatTensor([0] * shape[0])

    class FakeTensor(FakeFlatTensor):
        def numel(self):
            return len(self.values)

        def reshape(self, shape):
            assert shape == (-1,)
            return FakeFlatTensor(self.values)

    class FakeTorch:
        @staticmethod
        def cat(tensors, dim=0):
            assert dim == 0
            values = []
            for tensor in tensors:
                values.extend(tensor.values)
            return FakeFlatTensor(values)

    padded = _pad_tensor_to_group_size(FakeTensor([1, 2, 3]), 4, torch_module=FakeTorch)

    assert padded.values == (1, 2, 3, 0)


def test_allocate_quantized_buffer_uses_estimated_uint8_layout():
    class FakeTensor:
        device = "cuda:0"

        def numel(self):
            return 65

        def new_empty(self, shape, dtype):
            return {"shape": shape, "dtype": dtype, "device": self.device}

    class FakeTorch:
        uint8 = "uint8"

    output = allocate_quantized_buffer(FakeTensor(), CompressionConfig(), dtype="fp16", torch_module=FakeTorch)

    assert output == {"shape": (132,), "dtype": "uint8", "device": "cuda:0"}


def test_allocate_dequantized_buffer_pads_to_group_boundary():
    class FakeTensor:
        device = "cuda:0"
        dtype = "float16"

        def new_empty(self, shape, dtype):
            return {"shape": shape, "dtype": dtype, "device": self.device}

    output = allocate_dequantized_buffer(
        FakeTensor(),
        (65,),
        CompressionConfig(group_size=64),
        torch_module=object(),
    )

    assert output == {"shape": (128,), "dtype": "float16", "device": "cuda:0"}


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


def test_dequantize_tensor_can_enable_compact_kernel_layout():
    class Decoded:
        def reshape(self, shape):
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.ReduceOP = SimpleNamespace(NONE="none-enum")
            self.calls = []

        def dequantize(self, *args):
            self.calls.append(args)
            return Decoded()

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    buffer = object()

    dequantize_tensor(buffer, (2, 3), CompressionConfig(compact=True), dtype="fp16", extension_status=status)

    assert extension.calls == [(buffer, 64, 0, 8, "none-enum", "linear-enum", "fp16-enum", True)]


def test_dequantize_tensor_uses_inplace_extension_when_output_is_provided():
    class Output:
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

        def inplace_dequantize(self, *args):
            self.calls.append(args)

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    buffer = object()
    output = Output()

    result = dequantize_tensor(
        buffer,
        (2, 3),
        CompressionConfig(compact=True),
        dtype="fp16",
        extension_status=status,
        output=output,
    )

    assert result is output
    assert result.shape == (2, 3)
    assert extension.calls == [(buffer, output, 64, 0, 8, "none-enum", "linear-enum", True)]


def test_dequantize_tensor_can_sum_into_existing_output():
    class Output:
        def reshape(self, shape):
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.ReduceOP = SimpleNamespace(NONE="none-enum", SUM="sum-enum")
            self.calls = []

        def inplace_dequantize(self, *args):
            self.calls.append(args)

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    output = Output()

    dequantize_tensor(
        object(),
        (2, 3),
        CompressionConfig(),
        dtype="fp16",
        extension_status=status,
        output=output,
        reduce_op="sum",
    )

    assert extension.calls[0][5] == "sum-enum"


def test_dequantize_reduce_tensors_calls_extension_reduce_api():
    class Decoded:
        def __init__(self):
            self.shape = None

        def __truediv__(self, value):
            self.divisor = value
            return self

        def reshape(self, shape):
            self.shape = shape
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []
            self.decoded = Decoded()

        def dequantize_reduce(self, *args):
            self.calls.append(args)
            return self.decoded

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    buffers = ["rank0", "rank1"]

    result = dequantize_reduce_tensors(
        buffers,
        (2, 3),
        CompressionConfig(compact=True),
        dtype="fp16",
        extension_status=status,
        reduce="mean",
    )

    assert result is extension.decoded
    assert result.divisor == 2
    assert result.shape == (2, 3)
    assert extension.calls == [(buffers, 64, 0, 8, "linear-enum", "fp16-enum", True)]


def test_dequantize_tensor_trims_padded_output_before_reshape():
    class Decoded:
        def __init__(self, values):
            self.values = tuple(values)
            self.shape = None

        def reshape(self, shape):
            if shape == (-1,):
                return self
            self.shape = shape
            return self

        def __getitem__(self, item):
            return Decoded(self.values[item])

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.ReduceOP = SimpleNamespace(NONE="none-enum")
            self.decoded = Decoded(range(64))

        def dequantize(self, *args):
            return self.decoded

    status = CudaExtensionStatus(available=True, module=FakeExtension())

    result = dequantize_tensor(object(), (3, 4), CompressionConfig(group_size=64), dtype="fp16", extension_status=status)

    assert result.values == tuple(range(12))
    assert result.shape == (3, 4)


def test_quantization_codec_rejects_extension_without_required_symbols():
    status = CudaExtensionStatus(available=True, module=object())

    with pytest.raises(CCDLUnavailableError, match="missing required symbol"):
        quantize_tensor(object(), CompressionConfig(), extension_status=status)


def test_update_error_feedback_residual_calls_native_inplace_symbol():
    class FakeExtension:
        def __init__(self):
            self.calls = []

        def inplace_error_feedback_update(self, *args):
            self.calls.append(args)

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)

    result = update_error_feedback_residual(
        "prepared",
        "restored",
        "residual",
        extension_status=status,
    )

    assert result == "residual"
    assert extension.calls == [("prepared", "restored", "residual")]


def test_update_error_feedback_residual_rejects_missing_native_symbol():
    status = CudaExtensionStatus(available=True, module=object())

    with pytest.raises(CCDLUnavailableError, match="inplace_error_feedback_update"):
        update_error_feedback_residual(
            "prepared",
            "restored",
            "residual",
            extension_status=status,
        )
