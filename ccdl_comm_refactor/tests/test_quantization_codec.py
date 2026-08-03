from pathlib import Path
from types import SimpleNamespace

import pytest

from ccdl_comm.config import CompressionConfig
from ccdl_comm.cuda.loader import CudaExtensionStatus
from ccdl_comm.quantization.codec import (
    CCDLUnavailableError,
    _pad_tensor_to_group_size,
    allocate_dequantized_buffer,
    allocate_quantized_buffer,
    dequantize_reduce_tensors,
    dequantize_reduce_update_error_feedback,
    dequantize_tensor,
    inplace_dequantize_reduce_mean,
    inplace_dequantize_reduce_mean_update_error_feedback,
    inplace_quantize_pack,
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


def test_inplace_quantize_pack_forwards_residual_and_records_layout_metadata():
    class FakeTensor:
        def numel(self):
            return 65

    class FakeExtension:
        def __init__(self):
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_quantize_pack(self, *args):
            self.calls.append(args)
            return True

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    tensor = FakeTensor()
    output = object()
    residual = object()
    metadata = {}
    config = CompressionConfig(compact=True)

    assert inplace_quantize_pack(
        tensor,
        output,
        residual,
        config,
        metadata,
        extension_status=status,
    )
    assert extension.calls == [
        (tensor, output, residual, 64, 0, False, 8, "linear-enum", True),
    ]
    assert metadata == {
        "original_numel": 65,
        "padded_numel": 128,
        "padding_numel": 63,
        "fused_quant_pack": True,
    }


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


def test_inplace_dequantize_reduce_mean_forwards_native_abi_and_preserves_output_identity():
    class FakeExtension:
        def __init__(self):
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_dequantize_reduce_mean(self, *args):
            self.calls.append(args)
            return True

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    output = object()

    result = inplace_dequantize_reduce_mean(
        ["rank0", "rank1", "rank2"],
        output,
        CompressionConfig(compact=True),
        extension_status=status,
        reduce="mean",
    )

    assert result is True
    assert extension.calls == [
        (["rank0", "rank1", "rank2"], output, 64, 0, 8, "linear-enum", True, 3),
    ]


def test_inplace_dequantize_reduce_mean_passes_one_for_sum_and_rejects_invalid_reduce():
    class FakeExtension:
        QuantType = SimpleNamespace(Linear="linear-enum")

        def __init__(self):
            self.calls = []

        def inplace_dequantize_reduce_mean(self, *args):
            self.calls.append(args)
            return False

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)

    assert not inplace_dequantize_reduce_mean(
        ["rank0", "rank1"],
        "output",
        CompressionConfig(),
        extension_status=status,
        reduce="sum",
    )
    assert extension.calls[0][-1] == 1
    with pytest.raises(ValueError, match="unsupported dequantize-reduce mode"):
        inplace_dequantize_reduce_mean(
            ["rank0"],
            "output",
            CompressionConfig(),
            extension_status=status,
            reduce="max",
        )


def test_dequantize_reduce_tensors_uses_fused_inplace_mean_before_fallback():
    class Output:
        def reshape(self, shape):
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_dequantize_reduce_mean(self, *args):
            self.calls.append(("fused", args))
            return True

        def inplace_dequantize_reduce(self, *args):
            raise AssertionError("fused mean should replace fallback")

    extension = FakeExtension()
    output = Output()
    result = dequantize_reduce_tensors(
        ["rank0", "rank1"],
        (2,),
        CompressionConfig(),
        dtype="fp16",
        extension_status=CudaExtensionStatus(available=True, module=extension),
        output=output,
        reduce="mean",
    )

    assert result is output
    assert extension.calls == [("fused", (["rank0", "rank1"], output, 64, 0, 8, "linear-enum", False, 2))]


def test_dequantize_reduce_tensors_declined_fusion_divides_output_in_place():
    class Output:
        def __init__(self):
            self.divisors = []

        def div_(self, divisor):
            self.divisors.append(divisor)
            return self

        def __truediv__(self, divisor):
            raise AssertionError("fallback mean must preserve caller output with div_")

        def reshape(self, shape):
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_dequantize_reduce_mean(self, *args):
            self.calls.append(("fused", args))
            return False

        def inplace_dequantize_reduce(self, *args):
            self.calls.append(("fallback", args))

    extension = FakeExtension()
    output = Output()
    result = dequantize_reduce_tensors(
        ["rank0", "rank1"],
        (2,),
        CompressionConfig(),
        dtype="fp16",
        extension_status=CudaExtensionStatus(available=True, module=extension),
        output=output,
        reduce="mean",
    )

    assert result is output
    assert output.divisors == [2]
    assert [name for name, _ in extension.calls] == ["fused", "fallback"]


def test_dequantize_reduce_tensors_falls_back_when_extension_lacks_fused_symbol():
    class Output:
        def reshape(self, shape):
            return self

    class FakeExtension:
        def __init__(self):
            self.DType = SimpleNamespace(FP16="fp16-enum")
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_dequantize_reduce(self, *args):
            self.calls.append(args)

    extension = FakeExtension()
    output = Output()

    assert dequantize_reduce_tensors(
        ["rank0"],
        (1,),
        CompressionConfig(),
        dtype="fp16",
        extension_status=CudaExtensionStatus(available=True, module=extension),
        output=output,
    ) is output
    assert extension.calls == [(["rank0"], output, 64, 0, 8, "linear-enum", False)]


def test_fused_reduced_shard_uses_output_cuda_device_guard_before_selecting_stream():
    kernel_source = (
        Path(__file__).resolve().parents[1]
        / "ccdl_comm"
        / "csrc"
        / "quantization"
        / "dequant_reduce_kernel.cu"
    ).read_text(encoding="utf-8")

    assert "#include <c10/cuda/CUDAGuard.h>" in kernel_source
    assert "c10::cuda::CUDAGuard device_guard(output.device());" in kernel_source


def test_fused_feedback_uses_restored_cuda_device_guard_before_selecting_stream():
    kernel_source = (
        Path(__file__).resolve().parents[1]
        / "ccdl_comm"
        / "csrc"
        / "quantization"
        / "dequant_reduce_kernel.cu"
    ).read_text(encoding="utf-8")
    feedback_source = kernel_source.split(
        "bool inplace_dequantize_reduce_mean_update_error_feedback(", 1
    )[1]

    guard = "c10::cuda::CUDAGuard device_guard(restored.device());"
    assert guard in feedback_source
    assert feedback_source.index(guard) < feedback_source.index(
        "cudaStream_t stream = get_current_cuda_stream();"
    )


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


def test_dequantize_reduce_update_error_feedback_calls_combined_native_api():
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
            self.calls = []
            self.decoded = Decoded()

        def dequantize_reduce_update_error_feedback(self, *args):
            self.calls.append(args)
            return self.decoded

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)

    result = dequantize_reduce_update_error_feedback(
        ["rank0", "rank1"],
        "prepared",
        "residual",
        (2, 3),
        CompressionConfig(compact=True),
        dtype="fp16",
        extension_status=status,
        reduce="mean",
    )

    assert result is extension.decoded
    assert result.shape == (2, 3)
    assert extension.calls == [
        (["rank0", "rank1"], "prepared", "residual", 64, 0, 8, "linear-enum", "fp16-enum", True, 2)
    ]


def test_dequantize_reduce_update_error_feedback_rejects_missing_native_symbol():
    class FakeExtension:
        DType = SimpleNamespace(FP16="fp16-enum")
        QuantType = SimpleNamespace(Linear="linear-enum")

    status = CudaExtensionStatus(available=True, module=FakeExtension())

    with pytest.raises(CCDLUnavailableError, match="dequantize_reduce_update_error_feedback"):
        dequantize_reduce_update_error_feedback(
            ["rank0"],
            "prepared",
            "residual",
            (1,),
            CompressionConfig(),
            dtype="fp16",
            extension_status=status,
        )


def test_inplace_dequantize_reduce_mean_update_error_feedback_calls_workspace_native_api():
    class FakeExtension:
        def __init__(self):
            self.QuantType = SimpleNamespace(Linear="linear-enum")
            self.calls = []

        def inplace_dequantize_reduce_mean_update_error_feedback(self, *args):
            self.calls.append(args)
            return True

    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)

    result = inplace_dequantize_reduce_mean_update_error_feedback(
        ["rank0", "rank1"],
        "prepared",
        "restored-workspace",
        "residual-workspace",
        CompressionConfig(compact=True),
        extension_status=status,
        reduce="mean",
    )

    assert result is True
    assert extension.calls == [
        (
            ["rank0", "rank1"],
            "prepared",
            "restored-workspace",
            "residual-workspace",
            64,
            0,
            8,
            "linear-enum",
            True,
            2,
        )
    ]


def test_inplace_dequantize_reduce_mean_update_error_feedback_rejects_invalid_reduce_mode():
    class FakeExtension:
        QuantType = SimpleNamespace(Linear="linear-enum")

        def inplace_dequantize_reduce_mean_update_error_feedback(self, *args):
            raise AssertionError("invalid reduce mode should be rejected before native call")

    status = CudaExtensionStatus(available=True, module=FakeExtension())

    with pytest.raises(ValueError, match="unsupported dequantize-reduce mode"):
        inplace_dequantize_reduce_mean_update_error_feedback(
            ["rank0"],
            "prepared",
            "restored",
            "residual",
            CompressionConfig(),
            extension_status=status,
            reduce="max",
        )
