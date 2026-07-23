#include <torch/extension.h>
#include <limits>

#ifdef CCDL_COMM_EXPERIMENTAL_ACLNN
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"
#include "aclnnop/aclnn_dynamic_block_quant.h"
#endif


namespace {

#ifdef CCDL_COMM_EXPERIMENTAL_ACLNN
constexpr int64_t ACLNN_INT8_DTYPE = 2;
#endif

pybind11::object make_quantized_payload(torch::Tensor quantized, torch::Tensor scales, int64_t original_numel) {
    pybind11::object namespace_type = pybind11::module_::import("types").attr("SimpleNamespace");
    return namespace_type(
        pybind11::arg("buffer") = quantized.reshape({-1}),
        pybind11::arg("scales") = scales.reshape({-1}).to(torch::kFloat16),
        pybind11::arg("original_numel") = original_numel);
}

#ifdef CCDL_COMM_EXPERIMENTAL_ACLNN
pybind11::object quantize_linear_int8_aclnn(torch::Tensor groups, int64_t group_size, int64_t original_numel) {
    auto quantized = torch::empty(groups.sizes(), groups.options().dtype(torch::kInt8));
    auto scales = torch::empty({groups.size(0)}, groups.options().dtype(torch::kFloat32));
    double min_scale = 0.0;
    char *round_mode = nullptr;
    int64_t dst_type = ACLNN_INT8_DTYPE;
    int64_t row_block_size = 1;
    int64_t col_block_size = group_size;
    EXEC_NPU_CMD(
        aclnnDynamicBlockQuant,
        groups,
        min_scale,
        round_mode,
        dst_type,
        row_block_size,
        col_block_size,
        quantized,
        scales);
    return make_quantized_payload(quantized, scales, original_numel);
}
#endif

pybind11::object quantize_linear_int8_torch_ops(torch::Tensor groups, int64_t original_numel) {
    auto max_abs = groups.abs().amax({1}, false);
    auto eps = torch::full_like(max_abs, std::numeric_limits<float>::epsilon());
    auto scales = torch::maximum(max_abs / 127.0, eps);
    auto quantized = torch::round(groups / scales.reshape({-1, 1})).clamp(-127, 127).to(torch::kInt8);
    return make_quantized_payload(quantized, scales, original_numel);
}

}  // namespace


pybind11::object quantize_linear_int8(torch::Tensor tensor, int64_t group_size) {
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    auto flat = tensor.reshape({-1});
    const auto original_numel = flat.numel();
    const auto remainder = original_numel % group_size;
    if (remainder != 0) {
        const auto padding = group_size - remainder;
        auto zeros = torch::zeros({padding}, flat.options());
        flat = torch::cat({flat, zeros}, 0);
    }

    auto groups = flat.reshape({-1, group_size}).to(torch::kFloat32);
#ifdef CCDL_COMM_EXPERIMENTAL_ACLNN
    if (groups.device().is_cpu()) {
        return quantize_linear_int8_torch_ops(groups, original_numel);
    }
    return quantize_linear_int8_aclnn(groups, group_size, original_numel);
#else
    return quantize_linear_int8_torch_ops(groups, original_numel);
#endif
}


torch::Tensor dequantize_linear_int8(
    torch::Tensor buffer,
    torch::Tensor scales,
    int64_t original_numel,
    std::vector<int64_t> shape,
    std::string dtype,
    int64_t group_size) {
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    auto groups = buffer.reshape({-1, group_size}).to(torch::kFloat32);
    auto restored = groups * scales.reshape({-1, 1}).to(torch::kFloat32);
    auto flat = restored.reshape({-1}).slice(0, 0, original_numel);
    if (dtype == "fp16") {
        return flat.to(torch::kFloat16).reshape(shape);
    }
    if (dtype == "bf16") {
        return flat.to(torch::kBFloat16).reshape(shape);
    }
    if (dtype == "fp32") {
        return flat.to(torch::kFloat32).reshape(shape);
    }
    TORCH_CHECK(false, "unsupported dtype for dequantize_linear_int8: ", dtype);
}
