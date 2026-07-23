#include <torch/extension.h>
#include <limits>


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
    auto max_abs = groups.abs().amax({1}, false);
    auto eps = torch::full_like(max_abs, std::numeric_limits<float>::epsilon());
    auto scales = torch::maximum(max_abs / 127.0, eps);
    auto quantized = torch::round(groups / scales.reshape({-1, 1})).clamp(-127, 127).to(torch::kInt8);

    pybind11::object namespace_type = pybind11::module_::import("types").attr("SimpleNamespace");
    return namespace_type(
        pybind11::arg("buffer") = quantized.reshape({-1}),
        pybind11::arg("scales") = scales,
        pybind11::arg("original_numel") = original_numel);
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
