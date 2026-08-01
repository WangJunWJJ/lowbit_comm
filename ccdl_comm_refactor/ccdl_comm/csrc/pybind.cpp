#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "quantization/quant_api.cuh"
#include "quantization/dequant_api.cuh"
#include "quantization/enum.cuh"


namespace py = pybind11;

namespace {

int64_t dequant_reduce_bytes_per_group(DType dtype, int64_t group_size, int64_t topk, int64_t bit) {
    int64_t bytes_per_group;
    if (dtype == DType::FP32) {
        bytes_per_group = group_size * bit / 8 + 4;
        if (topk == 1) bytes_per_group += 8;
        else if (topk == 2) bytes_per_group += 12;
    } else {
        bytes_per_group = group_size * bit / 8 + 2;
        if (topk == 1) bytes_per_group += 4;
        else if (topk == 2) bytes_per_group += 6;
    }
    return bytes_per_group;
}

torch::Dtype dequant_reduce_torch_dtype(DType dtype) {
    if (dtype == DType::FP16) return torch::kHalf;
    if (dtype == DType::BF16) return torch::kBFloat16;
    if (dtype == DType::FP32) return torch::kFloat32;
    return torch::kHalf;
}

}  // namespace

void inplace_dequantize_reduce(std::vector<torch::Tensor> inputs, torch::Tensor output, int64_t group_size, int64_t topk, int64_t bit, QuantType quant_type, bool compact) {
    TORCH_CHECK(!inputs.empty(), "inputs must not be empty");
    if (try_inplace_dequantize_reduce_fused(inputs, output, group_size, topk, bit, quant_type, compact)) {
        return;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        inplace_dequantize(
            inputs[i],
            output,
            group_size,
            topk,
            bit,
            i == 0 ? ReduceOP::NONE : ReduceOP::SUM,
            quant_type,
            compact
        );
    }
}

torch::Tensor dequantize_reduce(std::vector<torch::Tensor> inputs, int64_t group_size, int64_t topk, int64_t bit, QuantType quant_type, DType dtype, bool compact) {
    TORCH_CHECK(!inputs.empty(), "inputs must not be empty");
    TORCH_CHECK(inputs[0].dtype() == torch::kUInt8, "input must be uint8");
    int64_t bytes_per_group = dequant_reduce_bytes_per_group(dtype, group_size, topk, bit);
    int64_t num_groups = inputs[0].numel() / bytes_per_group;
    auto options = torch::TensorOptions().dtype(dequant_reduce_torch_dtype(dtype)).device(inputs[0].device());
    torch::Tensor output = torch::empty({num_groups * group_size}, options);
    inplace_dequantize_reduce(inputs, output, group_size, topk, bit, quant_type, compact);
    return output;
}

torch::Tensor dequantize_reduce_update_error_feedback(
    std::vector<torch::Tensor> inputs,
    torch::Tensor prepared,
    torch::Tensor residual,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    DType dtype,
    bool compact,
    int64_t divisor
) {
    TORCH_CHECK(!inputs.empty(), "inputs must not be empty");
    TORCH_CHECK(divisor > 0, "divisor must be > 0");
    int64_t bytes_per_group = dequant_reduce_bytes_per_group(dtype, group_size, topk, bit);
    int64_t num_groups = inputs[0].numel() / bytes_per_group;
    auto options = torch::TensorOptions().dtype(dequant_reduce_torch_dtype(dtype)).device(inputs[0].device());
    torch::Tensor restored = torch::empty({num_groups * group_size}, options);
    bool fused_updated = inplace_dequantize_reduce_mean_update_error_feedback(
        inputs,
        prepared,
        restored,
        residual,
        group_size,
        topk,
        bit,
        quant_type,
        compact,
        divisor
    );
    if (fused_updated) {
        return restored;
    }
    inplace_dequantize_reduce(inputs, restored, group_size, topk, bit, quant_type, compact);
    if (divisor != 1) {
        restored = restored / divisor;
    }
    inplace_error_feedback_update(prepared, restored, residual);
    return restored;
}

PYBIND11_MODULE(ccdl_cuda_ops, m) {
    m.def("quantize", &quantize);
    m.def("dequantize", &dequantize);
    m.def("inplace_quantize", &inplace_quantize);
    m.def("inplace_dequantize", &inplace_dequantize);
    m.def("dequantize_reduce", &dequantize_reduce);
    m.def("inplace_dequantize_reduce", &inplace_dequantize_reduce);
    m.def("inplace_error_feedback_update", &inplace_error_feedback_update);
    m.def("dequantize_reduce_update_error_feedback", &dequantize_reduce_update_error_feedback);
    m.def("inplace_dequantize_reduce_mean_update_error_feedback", &inplace_dequantize_reduce_mean_update_error_feedback);
    py::enum_<ReduceOP>(m, "ReduceOP")
        .value("SUM", ReduceOP::SUM)
        .value("NONE", ReduceOP::NONE)
        .value("MIN", ReduceOP::MIN)
        .value("MAX", ReduceOP::MAX)
        .export_values();
    py::enum_<QuantType>(m, "QuantType")
        .value("Linear", QuantType::Linear)
        .value("Normal", QuantType::Normal)
        .value("Uniform", QuantType::Uniform)
        .value("E3M0", QuantType::E3M0)
        .value("E2M1", QuantType::E2M1)
        .export_values();
    py::enum_<DType>(m, "DType")
        .value("FP16", DType::FP16)
        .value("BF16", DType::BF16)
        .value("FP32", DType::FP32)
        .export_values();
}
