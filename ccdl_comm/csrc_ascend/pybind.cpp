#include <torch/extension.h>


pybind11::object quantize_linear_int8(torch::Tensor tensor, int64_t group_size);
torch::Tensor dequantize_linear_int8(
    torch::Tensor buffer,
    torch::Tensor scales,
    int64_t original_numel,
    std::vector<int64_t> shape,
    std::string dtype,
    int64_t group_size);


PYBIND11_MODULE(ccdl_cann_ops, m) {
    m.def("quantize_linear_int8", &quantize_linear_int8);
    m.def("dequantize_linear_int8", &dequantize_linear_int8);
}
