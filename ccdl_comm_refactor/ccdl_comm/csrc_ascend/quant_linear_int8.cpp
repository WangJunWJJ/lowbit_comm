#include <torch/extension.h>


pybind11::object quantize_linear_int8(torch::Tensor tensor, int64_t group_size) {
    TORCH_CHECK(false, "ccdl_cann_ops.quantize_linear_int8 is not implemented yet");
}


torch::Tensor dequantize_linear_int8(
    torch::Tensor buffer,
    torch::Tensor scales,
    int64_t original_numel,
    std::vector<int64_t> shape,
    std::string dtype,
    int64_t group_size) {
    TORCH_CHECK(false, "ccdl_cann_ops.dequantize_linear_int8 is not implemented yet");
}
